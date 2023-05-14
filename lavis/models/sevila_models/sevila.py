"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import copy
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast, BertTokenizer

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration

@registry.register_model("sevila")
class SeViLA(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
    }

    def __init__( self, img_size=224, drop_path_rate=0,
        use_grad_checkpoint=False, vit_precision="fp16", freeze_vit=True,
        num_query_token=32, t5_model="google/flan-t5-xl", prompt="",
        max_txt_len=32, frame_num=8, answer_num=5, apply_lemmatizer=False, task='qa'):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        
        self.task = task
        
        # vision backbone
        self.visual_encoder, self.ln_vision, self.ln_vision_loc = self.init_vision_encoder_sevila(
            img_size, drop_path_rate, use_grad_checkpoint, vit_precision)

        # freeze ViT
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False         
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
            
        # text backbone
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
        t5_model, config=t5_config)

        # freeze T5
        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16() 

        # Q-Former for Answerer
        self.Qformer, self.query_tokens = self.init_Qformer(
        num_query_token, self.visual_encoder.num_features)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.num_query_token = num_query_token
        self.t5_proj = nn.Linear(
        self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)
        
        # Q-Former for Localizer
        if 'loc' in task:
            self.Qformer_loc, self.query_tokens_loc = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features)

            self.Qformer_loc.cls = None
            self.Qformer_loc.bert.embeddings.word_embeddings = None
            self.Qformer_loc.bert.embeddings.position_embeddings = None
            for layer in self.Qformer_loc.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.t5_proj_loc = nn.Linear(
            self.Qformer_loc.config.hidden_size, self.t5_model.config.hidden_size
            )
            
        self.max_txt_len = 77
        answer_id = [71, 272, 205, 309, 262] # A B C D E
        self.answer_id = answer_id[:answer_num]
        self.yes_id, self.no_id = 4273, 150
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        
        self.frame_num = frame_num
        self.ANS_MAP = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
        self.frame_prefix = ['Frame: ']
        self.vid_prefix = ['Frame {}: '.format(str(i+1)) for i in range(frame_num)]
        
        
        if 'freeze_qa' in task:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.query_tokens.requires_grad = False
            self.t5_proj.requires_grad = False

        if 'freeze_loc' in task:
            for name, param in self.Qformer_loc.named_parameters():
                param.requires_grad = False
            self.query_tokens_loc.requires_grad = False
            self.t5_proj_loc.requires_grad = False
            
    def forward(self, samples,
        use_nucleus_sampling=False,
        num_beams=5, max_length=30,
        min_length=1, top_p=0.9,
        repetition_penalty=1.0, length_penalty=1.0,
        num_captions=1, temperature=1,):

        image = samples["video"]
        
        b, t, c, w, h = image.shape     
        image = image.reshape(-1, c, w, h)
        image_embeds = self.visual_encoder(image) 
        _, n, _ = image_embeds.shape
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device) # bt n c
        
        # Localizer self-refinement
        if 'train_loc' in self.task:

            # ========= Generate pseudo labels by frozen answerer ============
            with torch.no_grad():
                
                image_embeds_, image_atts_ = image_embeds.detach().clone(), image_atts.detach().clone()
                image_embeds_ = self.ln_vision(image_embeds_)
                
                query_tokens_qa = self.query_tokens.expand(image_embeds_.shape[0], -1, -1)
                query_output_qa = self.Qformer.bert(
                    query_embeds=query_tokens_qa, encoder_hidden_states=image_embeds_,
                    encoder_attention_mask=image_atts_, return_dict=True)
                inputs_t5_qa = self.t5_proj(query_output_qa.last_hidden_state)
                atts_t5_qa = torch.ones(inputs_t5_qa.size()[:-1], dtype=torch.long).to(image.device)
                text_input_qa = samples['qa_input']
                answer = samples['qa_output']
                ans_idx = [self.ANS_MAP[a[-1]] for a in answer]

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    # Frame Prefix
                    frame_prefix = self.t5_tokenizer(
                        self.frame_prefix, padding="longest", add_special_tokens=False,
                        truncation=True, max_length=self.max_txt_len, return_tensors="pt",
                        ).to(image.device) # 
                    frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b*t, 0)
                    frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b*t, 0)
                    # Question, options input
                    input_tokens_qa = self.t5_tokenizer(
                        text_input_qa, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    input_ids_qa = torch.repeat_interleave(input_tokens_qa.input_ids, t, 0)
                    input_attention_mask_qa = torch.repeat_interleave(input_tokens_qa.attention_mask, t, 0)

                    # Output target
                    output_tokens_qa = self.t5_tokenizer(
                        answer, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    targets_qa = output_tokens_qa.input_ids.masked_fill(
                        output_tokens_qa.input_ids == self.t5_tokenizer.pad_token_id, -100)
                    output_tokens_mask_qa = torch.repeat_interleave(output_tokens_qa.attention_mask, t, dim=0)
                    targets_qa = torch.repeat_interleave(targets_qa, t, dim=0)
                    
                    # input for QA
                    frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)
                    inputs_embeds_qa = self.t5_model.encoder.embed_tokens(input_ids_qa)
                    inputs_embeds_qa = torch.cat([frame_predix_embed, inputs_t5_qa, inputs_embeds_qa], dim=1)
                    encoder_atts_qa = torch.cat([frame_prefix_mask, atts_t5_qa, input_attention_mask_qa], dim=1)

                    outputs_embed_qa = self.t5_model(
                        inputs_embeds=inputs_embeds_qa, attention_mask=encoder_atts_qa,
                        decoder_attention_mask=output_tokens_mask_qa, return_dict=True, labels=targets_qa)
                    pred_logits_qa = outputs_embed_qa.logits.detach()
                    pred_logits_qa = pred_logits_qa[:, 1, self.answer_id] # b*t, 5
                    pred_ans = torch.argmax(pred_logits_qa, dim=-1)
                    pred_ans = pred_ans.reshape(b, -1) # b, t
                    # print('pred_ans', pred_ans)
                    pseudo_label = []
                    for i, preds in enumerate(pred_ans):
                        for p in preds:
                            if p == ans_idx[i]:
                                pseudo_label.append('yes')
                            else:
                                pseudo_label.append('no')
            # ================================================================
                
            # ============== Train localizer with pseudo labels =================
            text_input_loc = samples['loc_input']
            query_tokens_loc = self.query_tokens_loc.expand(image_embeds.shape[0], -1, -1)
            image_embeds = self.ln_vision_loc(image_embeds)
            
            query_output_loc = self.Qformer_loc.bert(
                query_embeds=query_tokens_loc, encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts, return_dict=True) # bt, n, c
            inputs_t5_loc = self.t5_proj_loc(query_output_loc.last_hidden_state) # bt, n, c
            atts_t5_loc = torch.ones(inputs_t5_loc.size()[:-1], dtype=torch.long).to(image.device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                frame_prefix = self.t5_tokenizer(
                    self.frame_prefix, padding="longest", add_special_tokens=False,
                    truncation=True, max_length=self.max_txt_len, return_tensors="pt").to(image.device) 
                frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b*t, 0)
                frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b*t, 0)
                frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)

                input_tokens_loc = self.t5_tokenizer(
                    text_input_loc, padding="longest", truncation=True,
                    max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                input_ids_loc = torch.repeat_interleave(input_tokens_loc.input_ids, t, 0)
                input_attention_mask_loc = torch.repeat_interleave(input_tokens_loc.attention_mask, t, 0)
                inputs_embeds_loc = self.t5_model.encoder.embed_tokens(input_ids_loc)
                    
                inputs_embeds_loc = torch.cat([frame_predix_embed, inputs_t5_loc, inputs_embeds_loc], dim=1)
                encoder_atts_loc = torch.cat([frame_prefix_mask, atts_t5_loc, input_attention_mask_loc], dim=1)

                output_tokens_loc = self.t5_tokenizer(
                    pseudo_label, padding="longest", truncation=True,
                    max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                targets_loc = output_tokens_loc.input_ids.masked_fill(
                    output_tokens_loc.input_ids == self.t5_tokenizer.pad_token_id, -100)
                output_tokens_loc_mask = output_tokens_loc.attention_mask
                
                outputs_loc = self.t5_model(
                    inputs_embeds=inputs_embeds_loc, attention_mask=encoder_atts_loc,
                    decoder_attention_mask=output_tokens_loc_mask,
                    return_dict=True, labels=targets_loc)
                loss = outputs_loc.loss
                                
            return {"loss": loss}
        
        # Finetune answerer with localizer
        elif 'train_qa_with_loc' in self.task:
            # frame selection
            with torch.no_grad():
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device) # bt n c
                image_embeds_, image_atts_ = image_embeds.detach().clone(), image_atts.detach().clone()
                image_embeds_ = self.ln_vision_loc(image_embeds_)
            
                text_input_loc = samples['loc_input']
                query_tokens_loc = self.query_tokens_loc.expand(image_embeds_.shape[0], -1, -1)
                query_output_loc = self.Qformer_loc.bert(
                    query_embeds=query_tokens_loc, encoder_hidden_states=image_embeds_,
                    encoder_attention_mask=image_atts_, return_dict=True)
                inputs_t5_loc = self.t5_proj_loc(query_output_loc.last_hidden_state)

                atts_t5_loc = torch.ones(inputs_t5_loc.size()[:-1], dtype=torch.long).to(image.device)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):

                    frame_prefix = self.t5_tokenizer(
                        self.frame_prefix, padding="longest", add_special_tokens=False,
                        truncation=True, max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b*t, 0)
                    frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b*t, 0)
                    frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)
                    input_tokens_loc = self.t5_tokenizer(
                        text_input_loc, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    input_ids_loc = torch.repeat_interleave(input_tokens_loc.input_ids, t, 0)
                    input_attention_mask_loc = torch.repeat_interleave(input_tokens_loc.attention_mask, t, 0)
                    inputs_embeds_loc = self.t5_model.encoder.embed_tokens(input_ids_loc)              
                    inputs_embeds_loc = torch.cat([frame_predix_embed, inputs_t5_loc, inputs_embeds_loc], dim=1)
                    encoder_atts_loc = torch.cat([frame_prefix_mask, atts_t5_loc, input_attention_mask_loc], dim=1)
    
                    outputs_loc = self.t5_model.generate(
                        inputs_embeds=inputs_embeds_loc, attention_mask=encoder_atts_loc,
                        do_sample=use_nucleus_sampling, top_p=top_p, temperature=temperature, num_beams=1,
                        max_new_tokens=max_length, min_length=min_length, repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty, num_return_sequences=num_captions,
                        return_dict_in_generate=True, output_hidden_states=True, output_scores=True)
                            
                    pred_logits_loc = outputs_loc.scores[0]
                    loc_yes = pred_logits_loc[:, self.yes_id]
                    loc_yes = loc_yes.reshape(b, -1)
                    
            text_input_qa = samples['qa_input']
            answer = samples['qa_output'] # Option A ...
            select_frames_idx = torch.topk(loc_yes, self.frame_num, dim=-1).indices.tolist()
            sorted_frames_idx = []
            image_embeds = self.ln_vision(image_embeds)
            image_embeds = image_embeds.reshape(b, t, n, -1)
            for frames in select_frames_idx:
                sorted_frames_idx.append(sorted(frames))
            select_frames = []
            for i, fs in enumerate(sorted_frames_idx): 
                video = []
                for j, f in enumerate(fs):
                    video.append(image_embeds[i][f])
                video = torch.stack(video, dim=0) # 4, n , -1
                select_frames.append(video)
                    
            select_frames = torch.stack(select_frames, dim=0) # b 4, n , -1
            select_frames = select_frames.reshape(-1, select_frames.shape[-2], select_frames.shape[-1])
            image_atts = torch.ones(select_frames.size()[:-1], dtype=torch.long).to(image.device) # bt n c
            query_tokens_qa = self.query_tokens.expand(select_frames.shape[0], -1, -1)
            query_output_qa = self.Qformer.bert(
                query_embeds=query_tokens_qa, encoder_hidden_states=select_frames,
                encoder_attention_mask=image_atts, return_dict=True)
            inputs_t5_qa = self.t5_proj(query_output_qa.last_hidden_state)
            inputs_t5_qa = inputs_t5_qa.reshape(b, -1, inputs_t5_qa.shape[-2], inputs_t5_qa.shape[-1])
            atts_t5_qa = torch.ones(inputs_t5_qa.size()[:-1], dtype=torch.long).to(image.device)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):        
                vid_prefix = self.t5_tokenizer(
                    self.vid_prefix, padding="longest", add_special_tokens=False,
                    truncation=True, max_length=self.max_txt_len, return_tensors="pt",).to(image.device) # 
                vid_prefix_id = torch.repeat_interleave(vid_prefix.input_ids.unsqueeze(0), b, 0)
                vid_prefix_mask = torch.repeat_interleave(vid_prefix.attention_mask.unsqueeze(0), b, 0)
                vid_prefix_embed = self.t5_model.encoder.embed_tokens(vid_prefix_id) # b t n_word c
                        
                inputs_t5_qa = torch.cat([vid_prefix_embed, inputs_t5_qa], dim=2) # b, t, n_word + m, c
                atts_t5_qa = torch.cat([vid_prefix_mask, atts_t5_qa], dim=2) # b, t, n_word + m 
                inputs_t5_qa = inputs_t5_qa.reshape(b, -1, inputs_t5_qa.shape[-1])
                atts_t5_qa = atts_t5_qa.reshape(b, -1)
                        
                input_tokens_qa = self.t5_tokenizer(
                    text_input_qa, padding="longest", truncation=True,
                    max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                inputs_embeds_qa = self.t5_model.encoder.embed_tokens(input_tokens_qa.input_ids) 
                inputs_embeds_qa = torch.cat([inputs_t5_qa, inputs_embeds_qa], dim=1)
                encoder_atts_qa = torch.cat([atts_t5_qa, input_tokens_qa.attention_mask], dim=1)
                
                output_tokens_qa = self.t5_tokenizer(
                    answer, padding="longest", truncation=True,
                    max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                targets_qa = output_tokens_qa.input_ids.masked_fill(
                    output_tokens_qa.input_ids == self.t5_tokenizer.pad_token_id, -100)
                output_tokens_mask_qa = output_tokens_qa.attention_mask
                
                outputs_qa = self.t5_model(
                    inputs_embeds=inputs_embeds_qa, attention_mask=encoder_atts_qa,
                    decoder_attention_mask=output_tokens_mask_qa, return_dict=True, labels=targets_qa)
                loss = outputs_qa.loss
                
                return {"loss": loss}
        
        # finetune answerer with random frames
        elif 'loc' not in self.task or 'train_qa_wo_loc' in self.task:
            #pass
            query_tokens_qa = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            image_embeds = self.ln_vision(image_embeds)
            
            query_output_qa = self.Qformer.bert(
                query_embeds=query_tokens_qa, encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts, return_dict=True)
            inputs_t5_qa = self.t5_proj(query_output_qa.last_hidden_state)
            text_input_qa = samples['qa_input'] 
            answer = samples['qa_output'] 

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    # Frame Prefix
                if 'qa_vid' not in self.task:
                    atts_t5_qa = torch.ones(inputs_t5_qa.size()[:-1], dtype=torch.long).to(image.device) 
                    frame_prefix = self.t5_tokenizer(
                        self.frame_prefix, padding="longest", add_special_tokens=False,
                        truncation=True, max_length=self.max_txt_len,return_tensors="pt",
                        ).to(image.device) 
                    frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b*t, 0)
                    frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b*t, 0)
                    # Question, Options input
                    input_tokens_qa = self.t5_tokenizer(
                        text_input_qa, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    input_ids_qa = torch.repeat_interleave(input_tokens_qa.input_ids, t, 0)
                    input_attention_mask_qa = torch.repeat_interleave(input_tokens_qa.attention_mask, t, 0)

                    # Output target
                    output_tokens_qa = self.t5_tokenizer(
                        answer, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    targets_qa = output_tokens_qa.input_ids.masked_fill(
                        output_tokens_qa.input_ids == self.t5_tokenizer.pad_token_id, -100)
                    output_tokens_mask_qa = torch.repeat_interleave(output_tokens_qa.attention_mask, t, dim=0)
                    targets_qa = torch.repeat_interleave(targets_qa, t, dim=0)
                    
                    # input for QA
                    frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)
                    inputs_embeds_qa = self.t5_model.encoder.embed_tokens(input_ids_qa)
                    inputs_embeds_qa = torch.cat([frame_predix_embed, inputs_t5_qa, inputs_embeds_qa], dim=1)
                    encoder_atts_qa = torch.cat([frame_prefix_mask, atts_t5_qa, input_attention_mask_qa], dim=1)
                else:
                    vid_prefix = self.t5_tokenizer(
                        self.vid_prefix, padding="longest", add_special_tokens=False,
                        truncation=True, max_length=self.max_txt_len, return_tensors="pt",).to(image.device) # 
                    vid_prefix_id = torch.repeat_interleave(vid_prefix.input_ids.unsqueeze(0), b, 0)
                    vid_prefix_mask = torch.repeat_interleave(vid_prefix.attention_mask.unsqueeze(0), b, 0)
                    vid_prefix_embed = self.t5_model.encoder.embed_tokens(vid_prefix_id) # b t n_word c
                    
                    inputs_t5_qa = inputs_t5_qa.reshape(b, t, inputs_t5_qa.shape[-2], -1) # b, t, m ,c
                    atts_t5_qa = torch.ones(inputs_t5_qa.size()[:-1], dtype=torch.long).to(image.device)
                    
                    inputs_t5_qa = torch.cat([vid_prefix_embed, inputs_t5_qa], dim=2) # b, t, n_word + m, c
                    atts_t5_qa = torch.cat([vid_prefix_mask, atts_t5_qa], dim=2) # b, t, n_word + m 
                    inputs_t5_qa = inputs_t5_qa.reshape(b, -1, inputs_t5_qa.shape[-1])
                    atts_t5_qa = atts_t5_qa.reshape(b, -1)
                    
                    input_tokens_qa = self.t5_tokenizer(
                        text_input_qa, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    inputs_embeds_qa = self.t5_model.encoder.embed_tokens(input_tokens_qa.input_ids) 
                    inputs_embeds_qa = torch.cat([inputs_t5_qa, inputs_embeds_qa], dim=1)
                    encoder_atts_qa = torch.cat([atts_t5_qa, input_tokens_qa.attention_mask], dim=1)
                    
                    output_tokens_qa = self.t5_tokenizer(
                        answer, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    targets_qa = output_tokens_qa.input_ids.masked_fill(
                        output_tokens_qa.input_ids == self.t5_tokenizer.pad_token_id, -100)
                    output_tokens_mask_qa = output_tokens_qa.attention_mask

                outputs_qa = self.t5_model(
                    inputs_embeds=inputs_embeds_qa, attention_mask=encoder_atts_qa,
                    decoder_attention_mask=output_tokens_mask_qa, return_dict=True, labels=targets_qa)
                loss = outputs_qa.loss
                
                return {"loss": loss}
        

    @torch.no_grad()
    def generate(self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5, max_length=30,
        min_length=1, top_p=0.9,
        repetition_penalty=1.0, length_penalty=1.0,
        num_captions=1, temperature=1,):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        out = {}
        image, qid = samples["video"], samples['question_id']
        text_input_qa, answer = samples['qa_input'], samples['qa_output']
        
        # uniform sampling
        if 'loc' not in self.task or 'uni_eval' in self.task:
            b, t, c, w, h = image.shape        
            image = image.reshape(-1, c, w, h)
            with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
                image_embeds = self.ln_vision(self.visual_encoder(image)) # bt, n, c
            _, n, _ = image_embeds.shape
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device) # bt n c
            
            query_tokens_qa = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output_qa = self.Qformer.bert(
                query_embeds=query_tokens_qa, encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts, return_dict=True)
            inputs_t5_qa = self.t5_proj(query_output_qa.last_hidden_state)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    # Frame Prefix
                if 'vid' not in self.task: 
                    atts_t5_qa = torch.ones(inputs_t5_qa.size()[:-1], dtype=torch.long).to(image.device) 
                    frame_prefix = self.t5_tokenizer(
                        self.frame_prefix, padding="longest", add_special_tokens=False,
                        truncation=True, max_length=self.max_txt_len, return_tensors="pt",).to(image.device) # 
                    frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b*t, 0)
                    frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b*t, 0)
                    input_tokens_qa = self.t5_tokenizer(
                        text_input_qa, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    input_ids_qa = torch.repeat_interleave(input_tokens_qa.input_ids, t, 0)
                    input_attention_mask_qa = torch.repeat_interleave(input_tokens_qa.attention_mask, t, 0)
                                        
                    # input for QA
                    frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)
                    inputs_embeds_qa = self.t5_model.encoder.embed_tokens(input_ids_qa)
                    inputs_embeds_qa = torch.cat([frame_predix_embed, inputs_t5_qa, inputs_embeds_qa], dim=1)
                    encoder_atts_qa = torch.cat([frame_prefix_mask, atts_t5_qa, input_attention_mask_qa], dim=1)
                
                elif 'qa_vid' in self.task:
                    vid_prefix = self.t5_tokenizer(
                        self.vid_prefix, padding="longest", add_special_tokens=False,
                        truncation=True, max_length=self.max_txt_len, return_tensors="pt",).to(image.device) # 
                    vid_prefix_id = torch.repeat_interleave(vid_prefix.input_ids.unsqueeze(0), b, 0)
                    vid_prefix_mask = torch.repeat_interleave(vid_prefix.attention_mask.unsqueeze(0), b, 0)
                    vid_prefix_embed = self.t5_model.encoder.embed_tokens(vid_prefix_id) # b t n_word c
                    
                    inputs_t5_qa = inputs_t5_qa.reshape(b, t, inputs_t5_qa.shape[-2], -1) # b, t, m ,c
                    atts_t5_qa = torch.ones(inputs_t5_qa.size()[:-1], dtype=torch.long).to(image.device)
                    
                    inputs_t5_qa = torch.cat([vid_prefix_embed, inputs_t5_qa], dim=2) # b, t, n_word + m, c
                    atts_t5_qa = torch.cat([vid_prefix_mask, atts_t5_qa], dim=2) # b, t, n_word + m 
                    inputs_t5_qa = inputs_t5_qa.reshape(b, -1, inputs_t5_qa.shape[-1])
                    atts_t5_qa = atts_t5_qa.reshape(b, -1)
                    
                    input_tokens_qa = self.t5_tokenizer(
                        text_input_qa, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    inputs_embeds_qa = self.t5_model.encoder.embed_tokens(input_tokens_qa.input_ids)
                    inputs_embeds_qa = torch.cat([inputs_t5_qa, inputs_embeds_qa], dim=1)
                    encoder_atts_qa = torch.cat([atts_t5_qa, input_tokens_qa.attention_mask], dim=1)

                outputs_qa = self.t5_model.generate(
                    inputs_embeds=inputs_embeds_qa, attention_mask=encoder_atts_qa,
                    do_sample=use_nucleus_sampling, top_p=top_p,
                    temperature=temperature, num_beams=1,
                    max_new_tokens=max_length, min_length=min_length,
                    repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                    num_return_sequences=num_captions, return_dict_in_generate=True,
                    output_hidden_states=True, output_scores=True)
                try:
                    pred_logits_qa = outputs_qa.scores[1]
                except:
                    pred_logits_qa = outputs_qa.scores[0]
                pred_logits_qa = pred_logits_qa[:, self.answer_id] # b, 5
                pred_ans = torch.argmax(pred_logits_qa, dim=-1).cpu().tolist() 
        
        # inference with localizer             
        else:
            
            b, t, c, w, h = image.shape        
            image = image.reshape(-1, c, w, h)
            with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
                image_embeds = self.visual_encoder(image) # bt, n, c
                
            _, n, _ = image_embeds.shape
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device) # bt n c
            image_embeds_, image_atts_ = image_embeds.detach().clone(), image_atts.detach().clone()
            image_embeds_ = self.ln_vision_loc(image_embeds_)
            
            text_input_loc = samples['loc_input'] # Q + Prompt: Is this a good frame can answer the question?
            query_tokens_loc = self.query_tokens_loc.expand(image_embeds_.shape[0], -1, -1)
            query_output_loc = self.Qformer_loc.bert(
                query_embeds=query_tokens_loc, encoder_hidden_states=image_embeds_,
                encoder_attention_mask=image_atts_, return_dict=True)
            inputs_t5_loc = self.t5_proj_loc(query_output_loc.last_hidden_state)

            atts_t5_loc = torch.ones(inputs_t5_loc.size()[:-1], dtype=torch.long).to(image.device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):

                frame_prefix = self.t5_tokenizer(
                    self.frame_prefix, padding="longest", add_special_tokens=False,
                    truncation=True, max_length=self.max_txt_len, return_tensors="pt").to(image.device) # 
                #print('frame_prefix 1', frame_prefix.input_ids.shape) 8, 4
                frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b*t, 0)
                frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b*t, 0)
                frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)
                input_tokens_loc = self.t5_tokenizer(
                    text_input_loc, padding="longest", truncation=True,
                    max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                #print('input_ids_loc.input_ids', input_tokens_loc.input_ids)
                input_ids_loc = torch.repeat_interleave(input_tokens_loc.input_ids, t, 0)
                #print('input_ids_loc', input_ids_loc)
                input_attention_mask_loc = torch.repeat_interleave(input_tokens_loc.attention_mask, t, 0)
                inputs_embeds_loc = self.t5_model.encoder.embed_tokens(input_ids_loc)              
                inputs_embeds_loc = torch.cat([frame_predix_embed, inputs_t5_loc, inputs_embeds_loc], dim=1)
                encoder_atts_loc = torch.cat([frame_prefix_mask, atts_t5_loc, input_attention_mask_loc], dim=1)
    
                outputs_loc = self.t5_model.generate(
                    inputs_embeds=inputs_embeds_loc, attention_mask=encoder_atts_loc,
                    do_sample=use_nucleus_sampling, top_p=top_p, temperature=temperature, num_beams=1,
                    max_new_tokens=max_length, min_length=min_length, repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty, num_return_sequences=num_captions,
                    return_dict_in_generate=True, output_hidden_states=True, output_scores=True)
                        
                pred_logits_loc = outputs_loc.scores[0]
                loc_yes = pred_logits_loc[:, self.yes_id]
                loc_yes = loc_yes.reshape(b, -1)
                if 'qa_vid' in self.task:
                    select_frames_idx = torch.topk(loc_yes, self.frame_num, dim=-1).indices.tolist()
                    sorted_frames_idx = []
                    image_embeds = self.ln_vision(image_embeds)
                    image_embeds = image_embeds.reshape(b, t, n, -1)
                    for frames in select_frames_idx:
                        sorted_frames_idx.append(sorted(frames))
                    out['frame_idx'] = sorted_frames_idx
                    select_frames = []
                    for i, fs in enumerate(sorted_frames_idx): 
                        video = []
                        for j, f in enumerate(fs):
                            video.append(image_embeds[i][f])
                        video = torch.stack(video, dim=0)
                        select_frames.append(video)
                    
                    select_frames = torch.stack(select_frames, dim=0) # b 4, n , -1
                    select_frames = select_frames.reshape(-1, select_frames.shape[-2], select_frames.shape[-1])
                    image_atts = torch.ones(select_frames.size()[:-1], dtype=torch.long).to(image.device) # bt n c
                    query_tokens_qa = self.query_tokens.expand(select_frames.shape[0], -1, -1)
                    query_output_qa = self.Qformer.bert(
                        query_embeds=query_tokens_qa, encoder_hidden_states=select_frames,
                        encoder_attention_mask=image_atts, return_dict=True)
                    inputs_t5_qa = self.t5_proj(query_output_qa.last_hidden_state)
                    inputs_t5_qa = inputs_t5_qa.reshape(b, -1, inputs_t5_qa.shape[-2], inputs_t5_qa.shape[-1])
                    atts_t5_qa = torch.ones(inputs_t5_qa.size()[:-1], dtype=torch.long).to(image.device)
                    
                    vid_prefix = self.t5_tokenizer(
                        self.vid_prefix, padding="longest", add_special_tokens=False,
                        truncation=True, max_length=self.max_txt_len, return_tensors="pt",).to(image.device) # 
                    vid_prefix_id = torch.repeat_interleave(vid_prefix.input_ids.unsqueeze(0), b, 0)
                    vid_prefix_mask = torch.repeat_interleave(vid_prefix.attention_mask.unsqueeze(0), b, 0)
                    vid_prefix_embed = self.t5_model.encoder.embed_tokens(vid_prefix_id) # b t n_word c
                    
                    inputs_t5_qa = torch.cat([vid_prefix_embed, inputs_t5_qa], dim=2) # b, t, n_word + m, c
                    atts_t5_qa = torch.cat([vid_prefix_mask, atts_t5_qa], dim=2) # b, t, n_word + m 
                    inputs_t5_qa = inputs_t5_qa.reshape(b, -1, inputs_t5_qa.shape[-1])
                    atts_t5_qa = atts_t5_qa.reshape(b, -1)
                    
                    input_tokens_qa = self.t5_tokenizer(
                        text_input_qa, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    inputs_embeds_qa = self.t5_model.encoder.embed_tokens(input_tokens_qa.input_ids) 
                    inputs_embeds_qa = torch.cat([inputs_t5_qa, inputs_embeds_qa], dim=1)
                    encoder_atts_qa = torch.cat([atts_t5_qa, input_tokens_qa.attention_mask], dim=1)
                    
                else:
                    select_frames_idx = torch.argmax(loc_yes, -1)
                    select_frames = []
                    image_embeds = self.ln_vision(image_embeds)
                    image_embeds = image_embeds.reshape(b, t, n, -1)
                    for i, f in enumerate(select_frames_idx):
                        select_frames.append(image_embeds[i][f])
                        
                    select_frames = torch.stack(select_frames, dim=0)
                    image_atts = torch.ones(select_frames.size()[:-1], dtype=torch.long).to(image.device) # bt n c
                    query_tokens_qa = self.query_tokens.expand(select_frames.shape[0], -1, -1)
                    query_output_qa = self.Qformer.bert(
                        query_embeds=query_tokens_qa, encoder_hidden_states=select_frames,
                        encoder_attention_mask=image_atts, return_dict=True)
                    inputs_t5_qa = self.t5_proj(query_output_qa.last_hidden_state)
                    atts_t5_qa = torch.ones(inputs_t5_qa.size()[:-1], dtype=torch.long).to(image.device)
                    
                    frame_prefix = self.t5_tokenizer(
                        self.frame_prefix, padding="longest", add_special_tokens=False, 
                        truncation=True, max_length=self.max_txt_len, return_tensors="pt").to(image.device) # 
                    frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b, 0)
                    frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b, 0)

                    input_tokens_qa = self.t5_tokenizer(
                        text_input_qa, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)

                    frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)
                    inputs_embeds_qa = self.t5_model.encoder.embed_tokens(input_tokens_qa.input_ids)

                    inputs_embeds_qa = torch.cat([frame_predix_embed, inputs_t5_qa, inputs_embeds_qa], dim=1)
                    encoder_atts_qa = torch.cat([frame_prefix_mask, atts_t5_qa, input_tokens_qa.attention_mask], dim=1)
                            
                outputs_qa = self.t5_model.generate(
                    inputs_embeds=inputs_embeds_qa, attention_mask=encoder_atts_qa,
                    do_sample=use_nucleus_sampling, top_p=top_p,
                    temperature=temperature, num_beams=1,
                    max_new_tokens=max_length, min_length=min_length,
                    repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                    num_return_sequences=num_captions, return_dict_in_generate=True,
                    output_hidden_states=True, output_scores=True)
                pred_logits_qa = outputs_qa.scores[1]
                pred_logits_qa = pred_logits_qa[:, self.answer_id] # b, 5
                pred_ans = torch.argmax(pred_logits_qa, dim=-1).cpu().tolist()
        
        out['output_text'] = pred_ans
        if 'qa_vid' not in self.task: 
            out['temp_idx'] = [j for i in range(b) for j in range(t)]
            out['answer'] = [a for a in answer for i in range(t)]
            out['qid'] = [q for q in qid for i in range(t)]
        else:
            out['answer'] = answer
            out['qid'] = qid

        return out
    
    @torch.no_grad()
    def generate_demo(self,
        video,
        text_input_qa,
        text_input_loc,
        keyframe_num,
        qid='demo',
        use_nucleus_sampling=False,
        num_beams=5, max_length=30,
        min_length=1, top_p=0.9,
        repetition_penalty=1.0, length_penalty=1.0,
        num_captions=1, temperature=1,):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        out = {}
        image, qid = video, qid
        text_input_qa, answer = text_input_qa, 0
        
        # inference with localizer             
            
        b, t, c, w, h = image.shape        
        image = image.reshape(-1, c, w, h)
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
            image_embeds = self.visual_encoder(image) # bt, n, c
                
        _, n, _ = image_embeds.shape
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device) # bt n c
        image_embeds_, image_atts_ = image_embeds.detach().clone(), image_atts.detach().clone()
        image_embeds_ = self.ln_vision_loc(image_embeds_)
            
        text_input_loc = text_input_loc # Q + Prompt: Is this a good frame can answer the question?
        query_tokens_loc = self.query_tokens_loc.expand(image_embeds_.shape[0], -1, -1)
        query_output_loc = self.Qformer_loc.bert(
            query_embeds=query_tokens_loc, encoder_hidden_states=image_embeds_,
            encoder_attention_mask=image_atts_, return_dict=True)
        inputs_t5_loc = self.t5_proj_loc(query_output_loc.last_hidden_state)

        atts_t5_loc = torch.ones(inputs_t5_loc.size()[:-1], dtype=torch.long).to(image.device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):

            frame_prefix = self.t5_tokenizer(
                self.frame_prefix, padding="longest", add_special_tokens=False,
                truncation=True, max_length=self.max_txt_len, return_tensors="pt").to(image.device) # 
                #print('frame_prefix 1', frame_prefix.input_ids.shape) 8, 4
            frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b*t, 0)
            frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b*t, 0)
            frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)
            input_tokens_loc = self.t5_tokenizer(
                text_input_loc, padding="longest", truncation=True,
                max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                #print('input_ids_loc.input_ids', input_tokens_loc.input_ids)
            input_ids_loc = torch.repeat_interleave(input_tokens_loc.input_ids, t, 0)
                #print('input_ids_loc', input_ids_loc)
            input_attention_mask_loc = torch.repeat_interleave(input_tokens_loc.attention_mask, t, 0)
            inputs_embeds_loc = self.t5_model.encoder.embed_tokens(input_ids_loc)              
            inputs_embeds_loc = torch.cat([frame_predix_embed, inputs_t5_loc, inputs_embeds_loc], dim=1)
            encoder_atts_loc = torch.cat([frame_prefix_mask, atts_t5_loc, input_attention_mask_loc], dim=1)
    
            outputs_loc = self.t5_model.generate(
                inputs_embeds=inputs_embeds_loc, attention_mask=encoder_atts_loc,
                do_sample=use_nucleus_sampling, top_p=top_p, temperature=temperature, num_beams=1,
                max_new_tokens=max_length, min_length=min_length, repetition_penalty=repetition_penalty,
                length_penalty=length_penalty, num_return_sequences=num_captions,
                return_dict_in_generate=True, output_hidden_states=True, output_scores=True)
                        
            pred_logits_loc = outputs_loc.scores[0]
            loc_yes = pred_logits_loc[:, self.yes_id]
            loc_yes = loc_yes.reshape(b, -1)
            if 'qa_vid' in self.task:
                select_frames_idx = torch.topk(loc_yes, keyframe_num, dim=-1).indices.tolist()
                sorted_frames_idx = []
                image_embeds = self.ln_vision(image_embeds)
                image_embeds = image_embeds.reshape(b, t, n, -1)
                for frames in select_frames_idx:
                    sorted_frames_idx.append(sorted(frames))
                out['frame_idx'] = sorted_frames_idx
                select_frames = []
                for i, fs in enumerate(sorted_frames_idx): 
                    video = []
                    for j, f in enumerate(fs):
                        video.append(image_embeds[i][f])
                    video = torch.stack(video, dim=0)
                    select_frames.append(video)
                    
                select_frames = torch.stack(select_frames, dim=0) # b 4, n , -1
                select_frames = select_frames.reshape(-1, select_frames.shape[-2], select_frames.shape[-1])
                image_atts = torch.ones(select_frames.size()[:-1], dtype=torch.long).to(image.device) # bt n c
                query_tokens_qa = self.query_tokens.expand(select_frames.shape[0], -1, -1)
                query_output_qa = self.Qformer.bert(
                    query_embeds=query_tokens_qa, encoder_hidden_states=select_frames,
                    encoder_attention_mask=image_atts, return_dict=True)
                inputs_t5_qa = self.t5_proj(query_output_qa.last_hidden_state)
                inputs_t5_qa = inputs_t5_qa.reshape(b, -1, inputs_t5_qa.shape[-2], inputs_t5_qa.shape[-1])
                atts_t5_qa = torch.ones(inputs_t5_qa.size()[:-1], dtype=torch.long).to(image.device)
                
                vid_prefix = self.t5_tokenizer(
                        self.vid_prefix, padding="longest", add_special_tokens=False,
                        truncation=True, max_length=self.max_txt_len, return_tensors="pt",).to(image.device) # 
                vid_prefix_id = torch.repeat_interleave(vid_prefix.input_ids.unsqueeze(0), b, 0)
                vid_prefix_mask = torch.repeat_interleave(vid_prefix.attention_mask.unsqueeze(0), b, 0)
                vid_prefix_embed = self.t5_model.encoder.embed_tokens(vid_prefix_id) # b t n_word c
                    
                inputs_t5_qa = torch.cat([vid_prefix_embed, inputs_t5_qa], dim=2) # b, t, n_word + m, c
                atts_t5_qa = torch.cat([vid_prefix_mask, atts_t5_qa], dim=2) # b, t, n_word + m 
                inputs_t5_qa = inputs_t5_qa.reshape(b, -1, inputs_t5_qa.shape[-1])
                atts_t5_qa = atts_t5_qa.reshape(b, -1)
                    
                input_tokens_qa = self.t5_tokenizer(
                        text_input_qa, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                inputs_embeds_qa = self.t5_model.encoder.embed_tokens(input_tokens_qa.input_ids) 
                inputs_embeds_qa = torch.cat([inputs_t5_qa, inputs_embeds_qa], dim=1)
                encoder_atts_qa = torch.cat([atts_t5_qa, input_tokens_qa.attention_mask], dim=1)
                    
            else:
                select_frames_idx = torch.argmax(loc_yes, -1)
                select_frames = []
                image_embeds = self.ln_vision(image_embeds)
                image_embeds = image_embeds.reshape(b, t, n, -1)
                for i, f in enumerate(select_frames_idx):
                    select_frames.append(image_embeds[i][f])
                        
                select_frames = torch.stack(select_frames, dim=0)
                image_atts = torch.ones(select_frames.size()[:-1], dtype=torch.long).to(image.device) # bt n c
                query_tokens_qa = self.query_tokens.expand(select_frames.shape[0], -1, -1)
                query_output_qa = self.Qformer.bert(
                    query_embeds=query_tokens_qa, encoder_hidden_states=select_frames,
                        encoder_attention_mask=image_atts, return_dict=True)
                inputs_t5_qa = self.t5_proj(query_output_qa.last_hidden_state)
                atts_t5_qa = torch.ones(inputs_t5_qa.size()[:-1], dtype=torch.long).to(image.device)
                    
                frame_prefix = self.t5_tokenizer(
                        self.frame_prefix, padding="longest", add_special_tokens=False, 
                        truncation=True, max_length=self.max_txt_len, return_tensors="pt").to(image.device) # 
                frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b, 0)
                frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b, 0)

                input_tokens_qa = self.t5_tokenizer(
                    text_input_qa, padding="longest", truncation=True,
                    max_length=self.max_txt_len, return_tensors="pt").to(image.device)

                frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)
                inputs_embeds_qa = self.t5_model.encoder.embed_tokens(input_tokens_qa.input_ids)

                inputs_embeds_qa = torch.cat([frame_predix_embed, inputs_t5_qa, inputs_embeds_qa], dim=1)
                encoder_atts_qa = torch.cat([frame_prefix_mask, atts_t5_qa, input_tokens_qa.attention_mask], dim=1)
                            
            outputs_qa = self.t5_model.generate(
                    inputs_embeds=inputs_embeds_qa, attention_mask=encoder_atts_qa,
                    do_sample=use_nucleus_sampling, top_p=top_p,
                    temperature=temperature, num_beams=1,
                    max_new_tokens=max_length, min_length=min_length,
                    repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                    num_return_sequences=num_captions, return_dict_in_generate=True,
                    output_hidden_states=True, output_scores=True)
            pred_logits_qa = outputs_qa.scores[1]
            pred_logits_qa = pred_logits_qa[:, self.answer_id] # b, 5
            pred_ans = torch.argmax(pred_logits_qa, dim=-1).cpu().tolist()
        
        out['output_text'] = pred_ans
        if 'qa_vid' not in self.task: 
            out['temp_idx'] = [j for i in range(b) for j in range(t)]
            # out['answer'] = [a for a in answer for i in range(t)]
            out['qid'] = [q for q in qid for i in range(t)]
        else:
            # out['answer'] = answer
            out['qid'] = qid

        return out

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
    ):
        image = samples["image"]
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]
        if prompt:
            text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        input_tokens = self.t5_tokenizer(
            text_input, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        frame_num = cfg.get("frame_num", 8)
        answer_num = cfg.get("answer_num", 5) 
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        task = cfg.get("task", 'train_loc_freeze_qa')

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            frame_num=frame_num,
            answer_num=answer_num,
            task=task,
        )
        model.load_checkpoint_from_config(cfg)
        # for sevila with qvh pretraining
        # need load blip-2 q-former ckpt to q-former_loc
        if 'loc' in task and 'qvh' not in task:
           model.load_qformer_loc()

        return model
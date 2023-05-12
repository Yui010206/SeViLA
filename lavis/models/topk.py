# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""

import math
import torch
import copy
import einops
import torch.nn.functional as F
from torch import nn

from dataclasses import dataclass
from typing import Optional
from enum import IntEnum
from einops import rearrange

class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 1000):
        super(PerturbedTopK, self).__init__()
        self.num_samples = num_samples
        self.k = k

    def __call__(self, x, sigma):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, sigma)


class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
        #print('x', x.shape)
        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
        perturbed_x = x[:, None, :] + noise * sigma # b, nS, d
        #print('perturbed_x', perturbed_x.shape)
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        #print('topk_results',topk_results)

        indices = topk_results.indices # b, nS, k
        indices = torch.sort(indices, dim=-1).values # b, nS, k
        # print('indices', indices.shape ,indices[0,0,0])

        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1) # b, k, d
        # print('perturbed_output', perturbed_output.shape, perturbed_output[0,indices[0,0,0],0,0])

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise
        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        if ctx.sigma <= 1e-20:
            b, _, k, d = ctx.perturbed_output.size()
            expected_gradient = torch.zeros(b, k, d).to(grad_output.device)
        else:
            expected_gradient = (
                torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
                / ctx.num_samples
                / (ctx.sigma)
            )

        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)

        return (grad_input,) + tuple([None] * 5)

def HardTopK(k, x):
    topk_results = torch.topk(x, k=k, dim=-1, sorted=False)
    indices = topk_results.indices # b, k
    indices = torch.sort(indices, dim=-1).values
    return indices


def batched_index_select(input, dim, index):
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def extract_frames_from_indices(x, indices):
    batch_size, _, n, channels = x.shape
    k = indices.shape[-1]
    all_frame = x
    frames = batched_index_select(all_frame, 1, indices)
    frames = frames.contiguous().view(batch_size, k, n, channels)
    return frames


def extract_frames_from_indicators(x, indicators):
    indicators = rearrange(indicators, "b d k -> b k d")
    frames = torch.einsum("b k d, b d n c-> b k n c",
                         indicators, x)
    return frames


class ModalityEmbeddingsID(IntEnum):
    TEXT_QUESTION = 0
    TEXT_EMBEDDING = 1
    TEXT_UNUSED = 2  # ignore
    VISUAL_EMBEDDING = 3
    VISUAL_UNUSED = 4  # ignore

class ModalityEmbeddings(nn.Module):
    """
    Provides embeddings that indicate type of modality; for use with multimodal inputs for ATP. See atp.py for usage.
    """
    def __init__(self,
                 d_model: int,
                 use_text_query: bool = False,
                 use_text_cands: bool = False,
                 n_cands: int = 5):
        """
        Details for each of these arguments are provided in ATPConfig.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=len(ModalityEmbeddingsID),
                                      embedding_dim=d_model)

        self.use_text_query = use_text_query
        self.use_text_cands = use_text_cands
        self.n_cands = n_cands if use_text_cands else 0
        self.n_text_feats = 1 if use_text_query else 0
        if use_text_cands:
            self.n_text_feats += n_cands

    def forward(self, x, num_frame):
        """
        x: torch.tensor of size (L, N, D)
        returns modality embeddings for x of size (L, *, D)
        """
        L, N, D = x.size()  # (sequence_length, batch_size, feature_dim)
        num_txt = L - num_frame
        
        # assemble the IDs for the modality encodings, language inputs then vision inputs
        class_ids = []
        if self.use_text_query:
            class_ids.extend([ModalityEmbeddingsID.TEXT_QUESTION,] * num_txt)
        # if self.use_text_cands:
        #     class_ids.extend([ModalityEmbeddingsID.TEXT_EMBEDDING,] * self.n_cands)
        class_ids.extend([ModalityEmbeddingsID.VISUAL_EMBEDDING,] * num_frame)
        
        class_ids = torch.tensor(
            class_ids,
            dtype=torch.long,
            device=x.device
        ).unsqueeze(-1)
        
        # return modality embeddings
        return self.embedding(class_ids)

@dataclass
class ATPConfig:
    '''
    ATPConfig contains the parameters needed for the ATPSelectorModel (and its ATPEncoder).
    '''
    # ATPEncoder params
    n_layers: int = 6
    n_heads: int = 4
    d_model: int = 256
    d_input_t: int = 2048
    d_input_v: int = 1408
    d_model_ff: int = 256
    enc_dropout: float = 0.1
    use_text_query: bool = True  # at least one use_text_* needs to be true for ATP to be multimodal
    use_text_cands: bool = False  # ^ see above. (note: if both are false, ATP is vision-only)
    n_cands: int = 5  # only relevant when use_text_cands is set to true
    # ATPSelector params
    use_ste: bool = True  # controls type of selector during ATP training; see ATPSelectorModel.forward
    sel_dropout: float = 0.0
    d_input: int = 512  # size of the input vision-language embeddings (e.g. CLIP-ViT-B32 is size 512)
    
    def default_args(cls):
        return cls(n_layers = 6,
                   n_heads = 4,
                   d_model = 256,
                   d_input_t = 2048,
                   d_input_v = 1408,
                   d_model_ff = 256,
                   enc_dropout = 0.1,
                   use_text_query = True,
                   use_text_cands = False,
                   n_cands = 5,
                   use_ste = True,
                   sel_dropout = 0.0,
                   d_input = 512)

    @classmethod
    def from_args(cls, args):
        return cls(n_layers = args.n_layers,
                   n_heads = args.n_heads,
                   d_model = args.d_model,
                   d_model_ff = args.d_model_ff,
                   enc_dropout = args.enc_dropout,
                   use_text_query = args.use_text_query,
                   use_text_cands = args.use_text_cands,
                   n_cands = args.n_cands,
                   use_ste = args.use_ste,
                   sel_dropout = args.sel_dropout,
                   d_input = args.d_input)

class ATPEncoder(nn.Module):
    """
    The multimodal transformer encoder for the ATP model. For analysis purposes, the ATP encoder
    does not use any positional information (no positional encodings + transformer / self-attention)
    and is generally kept low-capacity. If the goal is raw accuracy (not analysis), you can relax these constraints.
    """
    def __init__(self, config: ATPConfig):
        """
        config: ATPConfig with parameters for the (transformer-based, atemporal) encoder for ATP.
        See ATPConfig documentation for details.
        """
        super().__init__()
        self.d_model = config.d_model

        self.dropout = nn.Dropout(p=config.enc_dropout)


        self.modality_encoding = ModalityEmbeddings(d_model=self.d_model,
                                                    use_text_query=config.use_text_query,
                                                    use_text_cands=config.use_text_cands,
                                                    n_cands=config.n_cands)
        
        atp_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model_ff,
            dropout=config.enc_dropout,
            activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(atp_encoder_layer, config.n_layers)

    def forward(self, x_inputs: torch.tensor, vis_L):
        """
        x_inputs: torch.tensor of shape (L, N, D)
        """
        L, N, D = x_inputs.size()  # (sequence_length, batch_size, d_model)
        assert D == self.d_model, "inputs dimension mismatch"
        x_encoded = x_inputs * math.sqrt(self.d_model)
        x_encoded += self.modality_encoding(x_encoded, vis_L)
        x_encoded = self.dropout(x_encoded)
        x_encoded = self.transformer_encoder(x_encoded)

        return x_encoded

class TopK_Selector(nn.Module):
    """
    The Atemporal Probe (ATP) selector model. Takes as input a sequence of image-language 
    encoding and outputs a (discrete) selection over the input frames, to help analyze 
    downstream discriminative video-language tasks.
    """
    
    def __init__(self, config=ATPConfig, num_select=4):
        """
        config: ATPConfig with parameters for initializing the ATPSelectorModel (and its encoder).
        See ATPConfig documentation for details.
        """
        super().__init__()
        self.config = config
        self.t_embedding = nn.Linear(config.d_input_t, config.d_input)
        self.v_embedding = nn.Linear(config.d_input_v, config.d_input)
        self.embedding = nn.Linear(config.d_input, config.d_model)
        self.atp_encoder = ATPEncoder(config)
        self.dropout = nn.Dropout(p=config.sel_dropout)
        self.logits = nn.Linear(config.d_model, 1)
        self.num_select = num_select
        self.sigma = 0.1

    def forward(self,
                x_vis, # [b, t, d]
                x_txt, # [b, n, d]
                **kwargs):
        """
        """
        x_vis_cls = x_vis[:, :, 0, :] # b t n c
        N, vis_L, D = x_vis_cls.size()  # (batch_size, sequence_length, feature_dimension)
        # embed the input sequence to the (smaller) model dimension (d_model) with modality encodings.
        x_vis_cls = self.v_embedding(self.dropout(x_vis_cls))
        x_txt = self.t_embedding(self.dropout(x_txt))
        x_inputs = []
        x_vis_cls = x_vis_cls.permute(1, 0, 2)
        x_inputs.append(x_txt.permute(1,0,2)) # (n, b, d)
        x_inputs.append(x_vis_cls)
        x_inputs = torch.cat(x_inputs, dim=0)
        x_encoded = self.embedding(self.dropout(x_inputs))
        x_atp_encoded = self.atp_encoder(x_encoded, vis_L)
        x_atp_encoded = x_atp_encoded.permute(1, 0, 2)
        x_encoded_v = x_atp_encoded[:, -vis_L: , :]
        # obtain selection scores (logits)
        x_logits = self.logits(self.dropout(x_encoded_v)).squeeze()
        #print('x_logits', x_logits.shape)

        if self.training:
            indices = PerturbedTopKFunction.apply(x_logits, self.num_select)
            #print('indices', indices.shape)
            indices = einops.rearrange(indices, "b k d -> b d k")

            if indices is not None:
                qa_frames = extract_frames_from_indicators(x_vis, indices)
            else:
                raise RuntimeError("Empty indices!")
        else:
            indices = HardTopK(self.num_select, x_logits)
            if indices is not None:
                qa_frames = extract_frames_from_indices(x_vis, indices)
            else:
                raise RuntimeError("Empty indices!")


        return qa_frames

if __name__ == "__main__":
    selector_config = ATPConfig.default_args

    Selector = TopK_Selector(num_select=4) #.eval()

    x_vis = torch.rand([2, 8, 257, 1408])
    x_txt = torch.rand([2, 68, 2048])

    out = Selector(x_vis, x_txt)
    print(out.shape)



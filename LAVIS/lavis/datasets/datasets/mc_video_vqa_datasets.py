"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import torch
from collections import OrderedDict

from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)
import random

class __DisplMixin:
    def displ_item(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        return OrderedDict(
            {"file": vpath, "question": ann["question"], "answer": ann["answer"]}
        )

ANS_MAPPING = {0:'A',1:'B',2:'C',3:'D',4:'E'}
# NextQA
class MCVideoQADataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def _load_auxiliary_mappings(self):
        pass
    
    def _get_answer_label(self, answer):
        if answer in self.class_labels:
            return self.class_labels[answer]
        else:
            return len(self.class_labels)

    def __getitem__(self, index):
        
        result = None
        while result is None:

            ann = self.annotation[index]
            qid = ann['qid'] 

            if 'QVHighlight' in qid:
                q = ann['query']
            else:
                q = ann['question']
            
            # set video clip if 'start'&'end' timestamp in data
            if 'start' in ann:
                start, end = float(ann['start']), float(ann['end'])
                clip = [start, end]
            else:
                clip = None       
            
            if 'VLEP' in qid:
                qa_prompt = 'Upon observing the provided frames, what is the most probable subsequent event?'
                events = 'Option A: ' + ann['a0'] + ' Option B: ' + ann['a1']
                qa_prompt = qa_prompt + ' ' + events
                loc_prompt = 'Does the information within the frame provide the necessary details to predict next event?'
                loc_prompt = qa_prompt + ' ' + loc_prompt
                answers = 'Option ' + ANS_MAPPING[int(ann['answer'])]
                duration = 1

            elif 'QVHighlight' in qid:
                duration = ann['duration']
                if 'relevant_windows' in ann: 
                    relevant_windows = ann['relevant_windows']
                else:
                    relevant_windows = None # for test
                pseudo_options = 'Option A: yes. Option B: no.'
                if q[-1] != '.':
                    q += '.'      
                loc_prompt = 'Question: ' + q +  ' ' + pseudo_options + ' Does the information within the frame provide the necessary details to accurately answer the given question?'
                qa_prompt = 'Considering the information presented in the frame, select the correct answer from the options.'
                
                
            else:
                prompt = 'Question: ' + q
                for j in range(ann['num_option']):
                    a = ann['a{}'.format(j)]
                    prompt += ' Option {}: '.format(ANS_MAPPING[j])
                    prompt += a
                hints = 'Options: ('
                #hints = 'Captions: ('
                for j in range(ann['num_option']):
                    ans = ann['a{}'.format(str(j))]
                    hints += ans
                    hints += ' '
                hints += ')'
                qa_prompt = prompt + ' Considering the information presented in the frame, select the correct answer from the options.'
                loc_prompt = 'Question: ' + q +  ' ' + hints + ' Does the information within the frame provide the necessary details to accurately answer the given question?'                
                answers = 'Option ' + ANS_MAPPING[int(ann['answer'])]
                duration = 1
            
            try:
                if 'VLEP' in qid:
                    video_id = ann['video']
                    if ':' in video_id:
                        # we set absolute path for vlep as it takes multiple video source
                        # you may change below paths to you own path
                        video_path = '/nas-hdd/shoubin/vlep_ytb_clips_tars/videos/vlep_ytb_clips/'
                    else:
                        video_id = video_id[:-3]
                        video_path = '/nas-hdd/shoubin/videos/tvqa/videos_3fps_with_audio/'
                    vpath = os.path.join(video_path, video_id + '.mp4')
                else:
                    vpath = os.path.join(self.vis_root, str(ann['video']) + '.mp4')   
                    
                frms, indices, fps = self.vis_processor(vpath, clip_proposal=clip)
                frms = frms.permute(1, 0, 2, 3)
                assert len(frms) == self.vis_processor.n_frms
                
                if 'QVHighlight' in qid: 
                    time_stamp = [float(idx/fps) for idx in indices]
                    answers = []
                    if relevant_windows is not None:
                        for t in time_stamp:
                            flag = False
                            for span in relevant_windows:
                                if t >= float(span[0]) and t<= float(span[1]):
                                    answers.append('yes')
                                    flag = True 
                                    break
                            if not flag:
                                answers.append('no') 
                    else:
                        for t in time_stamp:
                            answers.append('no') # for test
                            
                    answers = '_'.join(answers)
                              
                result = True
            except Exception as e:
                
                print(f"Error while read file idx")
                print("video is: {}".format(ann['video']))
                index = random.randint(0, len(self.annotation) - 1)
                
        return {
            "video": frms,
            "qa_input": qa_prompt,
            "loc_input": loc_prompt,
            "qa_output": answers,
            "question_id": qid,
            'duration': duration
        }

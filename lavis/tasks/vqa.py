"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os
import torch
import numpy as np
import random

import lavis.common.dist_utils as dist_utils
from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_eval import VQAEval
from lavis.tasks.base_task import BaseTask
from lavis.common.dist_utils import main_process

@registry.register_task("vqa")
class VQATask(BaseTask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        prompt="",
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt

        self.answer_list = None

        self.ques_files = dict()
        self.anno_files = dict()

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)
        prompt = run_cfg.get("prompt", "")

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for dataset in datasets.values():
            for split in dataset:
                if (
                    hasattr(dataset[split], "coco_fmt_qust_file")
                    and dataset[split].coco_fmt_qust_file is not None
                ):
                    self.ques_files[split] = dataset[split].coco_fmt_qust_file
                    self.anno_files[split] = dataset[split].coco_fmt_anno_file

                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    # if answer_list is not provided, then set it to None
                    pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets

    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        for answer, ques_id in zip(answers, question_id):
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"question_id": ques_id, "answer": answer})

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_vqa_result",
            remove_duplicate="question_id",
        )

        metrics = self._report_metrics(result_file=result_file, split=split_name)

        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Use official VQA evaluation script to report metrics.
        """
        metrics = {}

        if split in self.ques_files and split in self.anno_files:
            vqa = VQA(self.anno_files[split], self.ques_files[split])
            vqa_result = vqa.loadRes(
                resFile=result_file, quesFile=self.ques_files[split]
            )

            # create vqaEval object by taking vqa and vqaRes
            # n is precision of accuracy (number of places after decimal), default is 2
            vqa_scorer = VQAEval(vqa, vqa_result, n=2)
            logging.info("Start VQA evaluation.")
            vqa_scorer.evaluate()

            # print accuracies
            overall_acc = vqa_scorer.accuracy["overall"]
            metrics["agg_metrics"] = overall_acc

            logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
            logging.info("Per Answer Type Accuracy is the following:")

            for ans_type in vqa_scorer.accuracy["perAnswerType"]:
                logging.info(
                    "%s : %.02f"
                    % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
                )
                metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

            with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
            ) as f:
                f.write(json.dumps(metrics) + "\n")

        return metrics

@registry.register_task("gqa")
class GQATask(VQATask):
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["answer"]
        
        for answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": answer, "gt_ans": gt_answer})

        return pred_qa_pairs
        
    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        TODO: add other evaluation metrics for GQA
        """

        results = json.load(open(result_file, "r"))
        acc = []
        vqa_tool = VQAEval()

        for res in results:
            if res["gt_ans"] is None:
                # prepare test results for leaderboard evaluation
                self._save_result_leaderboard(results)
                return

            gt_ans = res["gt_ans"]
            pred = res["pred_ans"]

            if self.inference_method == "generate":
                pred = vqa_tool.processPunctuation(pred)
                pred = vqa_tool.processDigitArticle(pred)

            vqa_acc = 1 if pred == gt_ans else 0

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics
        

@registry.register_task("aok_vqa")
class AOKVQATask(VQATask):
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
        )

        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["direct_answers"]

        for pred_answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            pred_qa_pairs.append(
                {"question_id": ques_id, "pred_ans": pred_answer, "gt_ans": gt_answer}
            )

        return pred_qa_pairs

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Implementing accuracy computation for AOKVQA, see
        https://github.com/allenai/aokvqa/blob/main/evaluation/eval_predictions.py#L45 for details.
        """
        # TODO add evaluation for multi-choice

        results = json.load(open(result_file, "r"))
        acc = []

        for res in results:
            if res["gt_ans"] is None:
                # prepare test results for leaderboard evaluation
                self._save_result_leaderboard(results)
                return

            pred = res["pred_ans"]
            gt_ans = res["gt_ans"]

            num_match = sum([pred == gt for gt in gt_ans])
            vqa_acc = min(1.0, num_match / 3.0)

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics

    @dist_utils.main_process
    def _save_result_leaderboard(self, results):
        """
        Saving the results in the format required for leaderboard evaluation.

        [TODO] add support for multi-choice.
        """
        result_leaderboard = dict()
        for res in results:
            result_leaderboard[res["question_id"]] = {
                "direct_answer": res["pred_ans"],
                "multiple_choice": "",
            }

        result_file = registry.get_path("result_dir") + "_leaderboard.json"

        with open(result_file, "w") as f:
            json.dump(result_leaderboard, f)

        logging.info(f"Saved results for leaderboard evaluation at {result_file}")
    
@registry.register_task("frameqa")
class FrameQA(BaseTask):
    def __init__(self):
        super().__init__()
        self.ANS_MAPPING = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}

    def valid_step(self, model, samples):
        results = []

        outputs = model.generate(samples)

        answer = outputs["answer"]
        qid = outputs["qid"]
        output_text = outputs['output_text']
        temp_idx = outputs['temp_idx']
        assert len(qid)==len(temp_idx)
        assert len(qid)==len(output_text)
        assert len(qid)==len(answer) 
        
        for a, q, o, i in zip(answer, qid, output_text, temp_idx):
            # l =  l[self.ANS_MAPPING[a[-1]]]
            results.append(
                {
                    "qid": q,
                    'idx': i,
                    "prediction": o,
                    "target": self.ANS_MAPPING[a[-1]],
                }
            )

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch)
        )

        metrics = self._report_metrics(
            eval_result_file=eval_result_file, split_name=split_name
        )

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        results = json.load(open(eval_result_file))
        total_num = len(results)
        acc = 0
        group_by_qid = {}
        qtype_correct_dict = {}
        qtype_total_dict = {}
        for r in results:
            
            if r['qid'] not in group_by_qid:
                group_by_qid[r['qid']] = {} 
                group_by_qid[r['qid']]['idx'] = [r['idx']]
                group_by_qid[r['qid']]['pred'] = [r['prediction']]
                group_by_qid[r['qid']]['target'] = r['target']
            else:
                group_by_qid[r['qid']]['idx'].append(r['idx'])
                group_by_qid[r['qid']]['pred'].append(r['prediction'])
                
            qtype = r['qid'][0]
            if qtype not in qtype_total_dict:
                qtype_total_dict[qtype] = 1
            else:
                qtype_total_dict[qtype] += 1 

            if r['prediction'] == r['target']:
                acc += 1
                if qtype not in qtype_correct_dict:
                    qtype_correct_dict[qtype] = 1
                else:
                    qtype_correct_dict[qtype] += 1 
                
        oracle = 0 
        num = len(group_by_qid.keys())
        for q in group_by_qid:
            if group_by_qid[q]['target'] in group_by_qid[q]['pred']:
                oracle += 1
        
        metrics = {"agg_metrics": oracle/num , 'num': num, 'avg_acc': acc/total_num * 100, 'total':total_num}
        
        for qtype in qtype_total_dict:
            metrics[qtype] = qtype_correct_dict[qtype] / qtype_total_dict[qtype] * 100

        log_stats = {split_name: {k: v for k, v in metrics.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        logging.info(metrics)
        return metrics
    

@registry.register_task("videoqa")
class VideoQA(BaseTask):
    def __init__(self):
        super().__init__()
        self.ANS_MAPPING = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}

    def valid_step(self, model, samples):
        results = []

        outputs = model.generate(samples)

        answer = outputs["answer"]
        qid = outputs["qid"]
        output_text = outputs['output_text']
        if 'frame_idx' in outputs:
            frame_idx = outputs['frame_idx']
        else:
            frame_idx = [0 for i in range(len(qid))]
        # print(qid)
        # print(len(output_text), output_text)
        assert len(qid)==len(output_text)
        assert len(qid)==len(answer) 
        
        for a, q, o, f in zip(answer, qid, output_text, frame_idx):
            # l =  l[self.ANS_MAPPING[a[-1]]]
            results.append(
                {
                    "qid": q,
                    "prediction": o,
                    "target": self.ANS_MAPPING[a[-1]],
                    "frame_idx": f
                }
            )

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch)
        )

        metrics = self._report_metrics(
            eval_result_file=eval_result_file, split_name=split_name
        )

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        results = json.load(open(eval_result_file))
        total_num = len(results)
        acc = 0
        qtype_correct_dict = {}
        qtype_total_dict = {}
        for r in results:    
            qtype = r['qid'].split('_')[0]
            if qtype not in qtype_total_dict:
                qtype_total_dict[qtype] = 1
            else:
                qtype_total_dict[qtype] += 1 

            if r['prediction'] == r['target']:
                acc += 1
                if qtype not in qtype_correct_dict:
                    qtype_correct_dict[qtype] = 1
                else:
                    qtype_correct_dict[qtype] += 1 
        
        metrics = {"agg_metrics": acc/total_num , 'total':total_num}
        
        for qtype in qtype_total_dict:
            metrics[qtype] = qtype_correct_dict[qtype] / qtype_total_dict[qtype] * 100
            
        # for STAR
        if ('Interaction' in metrics) and ('Sequence' in metrics) and ('Prediction' in metrics) and ('Feasibility' in metrics):
            metrics["agg_metrics"] = (metrics['Interaction'] + metrics['Sequence'] + metrics['Prediction'] + metrics['Feasibility']) / 4

        log_stats = {split_name: {k: v for k, v in metrics.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        logging.info(metrics)
        return metrics
    
    
@registry.register_task("moment_retrieval")
class MR(BaseTask):
    def __init__(self):
        super().__init__()
        self.ANS_MAPPING = {'no': 0, 'yes': 1}

    def valid_step(self, model, samples):
        results = []

        outputs = model.generate(samples)
        answer = outputs['answer']
        qid = outputs['qid']
        score = outputs['yes_score']
        pred = outputs['pred_ans']
        assert len(qid)==len(answer)
        assert len(qid)==len(score)
        assert len(qid)==len(pred) 
        
        i = 0
        for a, q, s, p in zip(answer, qid, score, pred):
            # l =  l[self.ANS_MAPPING[a[-1]]]
            results.append(
                {
                    "qid": q + '_' + str(i),
                    "prediction": p,
                    "target": self.ANS_MAPPING[a],
                    'score': s
                }
            )
            i += 1

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch)
        )

        metrics = self._report_metrics(
            eval_result_file=eval_result_file, split_name=split_name
        )

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        results = json.load(open(eval_result_file))
        total_num = len(results)
        acc = 0
        for r in results:
            if r['prediction'] == r['target']:
                acc += 1
        metrics = {"agg_metrics": acc / total_num, 'total': total_num}
        log_stats = {split_name: {k: v for k, v in metrics.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        logging.info(metrics)
        return metrics
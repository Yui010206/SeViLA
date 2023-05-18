# Self-Chained Image-Language Model for Video Localization and Question Answering


## Dataset Preparation
We test our model on:
+ [NExT-QA](https://doc-doc.github.io/docs/nextqa.html)

+ [STAR](https://star.csail.mit.edu/)

+ [How2QA](https://value-benchmark.github.io/index.html)

+ [TVQA](https://tvqa.cs.unc.edu/)

+ [VLEP](https://value-benchmark.github.io/index.html)

+ [QVHighlights](https://github.com/jayleicn/moment_detr)

We re-format original json/csv/jsonl files in different dataset to the same json format via jupyter script.

Please set your own dataset/video path in running scripts or in dataset config files. For example:

* Option 1: change in running scripts

```bash
result_dir="YOUR_PATH"
train_path="YOUR_PATH"
val_path="YOUR_PATH"
video_path="YOUR_PATH"

exp_name='nextqa_infer'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 evaluate.py \
--cfg-path lavis/projects/sevila/eval/nextqa_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
datasets.nextqa.build_info.annotations.train.storage=${train_path} \
datasets.nextqa.build_info.annotations.val.storage=${val_path} \
datasets.nextqa.build_info.annotations.test.storage=${val_path} \
datasets.nextqa.build_info.videos.storage=${video_path} \
model.frame_num=4 \
datasets.nextqa.vis_processor.eval.n_frms=32 \
run.batch_size_eval=8 \
model.task='qvh_freeze_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'

```

* Option 2: change in dataset config file:

change [config files](../lavis/configs/datasets/nextqa/defaults_qa.yaml)






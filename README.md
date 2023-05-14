# Self-Chained Image-Language Model for Video Localization and Question Answering

* Authors: [Shoubin Yu](https://yui010206.github.io/), [Jaemin Cho](https://j-min.io), [Prateek Yadav](https://prateek-yadav.github.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)
* [arXiv](https://arxiv.org/abs/2305.06988)
<img src="./assets/teaser.png" alt="teaser image" width="800"/>

<img src="./assets/model.png" alt="teaser image" width="800"/>

<img src="./assets/chain.png" alt="teaser image" width="800"/>


# Code structure
```bash

# Data & Data Preprocessing
./sevila_data

# Pretrained Checkpoints
./sevila_checkpoints

# SeViLA code
./lavis/

# running scripts for SeViLa localizer/answerer training/inference
./run_scripts

```

# Setup

## Install Dependencies

1. (Optional) Creating conda environment

```bash
conda create -n sevila python=3.8
conda activate sevila
```

2. build from source

```bash
pip install -e .
```

## Download Pretrained Models
We pre-train SeViLA localizer on QVHighlights and hold checkpoints via [Huggingface](https://huggingface.co/Shoubin/SeViLA/resolve/main/sevila_pretrained.pth).
Download checkpoints and put it under /sevila_checkpoints.
The checkpoints (814.55M) contains pre-trained localizer and zero-shot answerer.



# Dataset Preparation
We test our model on:
+ [NExT-QA](https://doc-doc.github.io/docs/nextqa.html)

+ [STAR](https://star.csail.mit.edu/)

+ [How2QA](https://value-benchmark.github.io/index.html)

+ [TVQA](https://tvqa.cs.unc.edu/)

+ [VLEP](https://value-benchmark.github.io/index.html)

+ [QVHighlights](https://github.com/jayleicn/moment_detr)

please download original data and preprocess them via our [scripts](sevila_data/) under ./sevila_data/ . 


# Training and Inference
We provideo SeViLA training and inference script examples as following:
## 1) Localizer Pre-training
```bash
sh run_scripts/sevila/pre-train/pretrain_qvh.sh
```

## 2) Localizer Self-refinement

```bash
sh run_scripts/sevila/refinement/nextqa_sr.sh
```

## 3) Answerer Fine-tuning

```bash
sh run_scripts/sevila/finetune/nextqa_ft.sh
```

## 4) Inference

```bash
sh run_scripts/sevila/inference/nextqa_infer.sh
```


# Acknowledgments
We thank the developers of [LAVIS](https://github.com/salesforce/LAVIS), [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [CLIP](https://github.com/openai/CLIP), [All-in-one](https://github.com/showlab/all-in-one), for their public code release.


# Reference
Please cite our paper if you use our models in your works:


```bibtex
@misc{yu2023selfchained,
      title={Self-Chained Image-Language Model for Video Localization and Question Answering}, 
      author={Shoubin Yu and Jaemin Cho and Prateek Yadav and Mohit Bansal},
      year={2023},
      eprint={2305.06988},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

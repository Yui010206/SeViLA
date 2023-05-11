# Self-Chained Image-Language Model for Video Localization and Question Answering

* Authors: [Shoubin Yu](https://yui010206.github.io/), [Jaemin Cho](https://j-min.io), [Prateek Yadav](https://prateek-yadav.github.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)
* [arXiv]()
<img src="./assets/teaser.png" alt="teaser image" width="800"/>


# Code structure
```bash
# LAVIS library
./LAVIS/


# Data & Data Preprocessing
./data

# running scripts for SeViLa localizer/answerer training
./scripts

# Pretrained checkpoints
./checkpoints

```

# Setup

## Install Dependencies
Our work is based on Saleforce [LAVIS](https://github.com/salesforce/LAVIS) library, please refer to Installation section in LAVIS.


## Download Pretrained models
We pre-train SeViLA localizer on QVHighlights and hold checkpoints via [google drive](https://drive.google.com/file/d/17n7Y8IcwSqFfVu2BzIL58bF-F1HPK8cB/view?usp=sharing).
Download checkpoints and put it under /checkpoints.
The checkpoints (814.55M) contains pre-trained localizer and zero-shot answerer.



# Dataset preparation
We test our model on:
+ [NExT-QA](https://doc-doc.github.io/docs/nextqa.html)

+ [STAR](https://star.csail.mit.edu/)

+ [How2QA](https://value-benchmark.github.io/index.html)

+ [TVQA](https://tvqa.cs.unc.edu/)

+ [VLEP](https://value-benchmark.github.io/index.html)

+ [QVHighlights](https://github.com/jayleicn/moment_detr)

please download original data and preprocess them via our [scripts](data/). 


# Training and Inference
We provideo SeViLA training and inference scripts as following:
## 1) Localizer Pre-training
```bash
cd LAVIS

sh run_scripts/sevila/pre-train/pretrain_qvh.sh
```

## 2) Localizer Self-refinement

```bash
cd LAVIS

sh run_scripts/sevila/refinement/nextqa_sr.sh
```

## 3) Answerer Fine-tuning

```bash
cd LAVIS

sh run_scripts/sevila/finetune/nextqa_ft.sh
```

## 4) Inference

```bash
cd LAVIS

sh run_scripts/sevila/inference/nextqa_infer.sh
```


# Acknowledgments
We thank the developers of [LAVIS](https://github.com/clip-vil/CLIP-ViL/tree/master/CLIP-ViL-Direct/caption), [CLIP](https://github.com/openai/CLIP), [All-in-one](https://github.com/showlab/all-in-one), for their public code release.

# Multi-Feature Vision Transformer via Self-Supervised Learning for COVID-19 Diagnosis

## Introduction ##

This is the code to repoduce the study of [Multi-Feature Vision Transformer via Self-Supervised Representation Learning for Improvement of COVID-19 Diagnosis](arxiv update soon). Please cite our study if you are using this dataset or referring to our method.

## Network Structure ##

<img width="590" alt="NN_structure3" src="https://user-images.githubusercontent.com/31194584/181409364-27037733-80b3-4fe0-afad-1cbb36d95943.png">
# Multi-Feature-Semi-Supervised-Learning_COVID-19 (Pytorch)


## Result ##

- Test-1

Method | Labeled Sample (%) | Precision  | Recall | F1-Scores | Top-1(%)
------ | ------------------ | ---------- | ------ | --------- |-------- 
MF-ViT CA  | 10 | 0.91  | 0.91 | 0.91 | 91.10
MF-ViT CA  | 30 | 0.93  | 0.93 | 0.93 | 93.27
MF-ViT CA  | 100 | 0.95  | 0.95 | 0.95 | 95.03


## Usage ##

- Dataset and Trained model weights:
  - Download them from [Kaggle](https://www.kaggle.com/endiqq/largest-covid19-dataset?select=covid_metadata.csv). CXR folder are all origianl CXR images and Enh folder are all corresponding enhanced images. 

- Train MoCo-COVID:
  - python MoCo-COVID/moco_pretraining/moco/main_covid_mocov3based_single_img_type_5draws_mocov3structure_mocov2loss_vitsmall.py  -a vit_small -b 16 --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 --epochs=30 --warmup-epochs=4  --stop-grad-conv1 --moco-m-cos --moco-t=.2 --multiprocessing-distributed --world-size 1 --rank 0 --aug-setting chexpert --rotate 10 --exp-name  --train_data data --cos  (train_data: data=original CXR image; Train_Mix=enhanced CXR image)

- Finetune MoCo-COVID-LP:
 - python MoCo-CXR/moco_pretraining/moco/main_vit_covid_test_val_single_img_type_5draws_rev_v2loss_v3structure_vitsmall.py -a vit_small --lr 3 --batch-size 16 --epochs 90 --exp-name  --pretrained  --maintain-ratio --rotate --aug-setting chexpert --train_data data --optimizer sgd --cos (--pretraind = pretrained weights)

- Finetune MoCo-COVID-FT:
 - python MoCo-CXR/moco_pretraining/moco/main_vit_covid_test_val_single_img_type_5draws_rev_v2loss_v3structure_vitsmall.py -a vit_small --lr 3 --batch-size 16 --epochs 90 --exp-name  --pretrained  --maintain-ratio --rotate --aug-setting chexpert --train_data data --optimizer sgd --cos --semi-supervised (--pretraind = pretrained weights)



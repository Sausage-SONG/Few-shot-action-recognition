# Semi-supervised Few-shot Atomic Action Recognition

This repo contains the codes for our paper "Semi-supervised Few-shot Atomic Action Recognition". Please check our [paper](https://arxiv.org/abs/2011.08410) and [project page](https://sausage-song.github.io/home/FSAA/) for more details.

![FSAA Architecture](https://github.com/Sausage-SONG/Few-shot-action-recognition/raw/master/FSAA.jpg)

Our learning strategies are divided into two parts: 1) train an encoder with unsupervised learning; 2) train the action classification module with supervised learning. Regarding the encoder our model provides fine-grained spatial and temporal video processing with high length flexibility, which embeds the video feature and temporally combines the features with TCN. In terms of classification module, our models provides attention pooling and compares the multi-head relation. Finally, the CTC and MSE loss enables our model for time-invariant few shot classification training.

# Requirements

pytorch >= 1.5.0  
torchvision >= 0.6.0  
numpy >= 1.18.1  
scipy >= 1.4.1  
[vidaug](https://github.com/okankop/vidaug) >= 0.1

# Usage

## Installation

1. Clone the repo
2. Install [required packages](#requirements)
3. Download [trained models](#trained-models) to `<REPO_DIR>/models` (Optional)
4. Download the [datasets](#datasets) (Optional)

## Training

As mentioned in the [intro](#semi-supervised-few-shot-atomic-action-recognition), our model training has two parts.

1. Train the encoder unsupervisedly.  
   Here we use [MoCo](https://github.com/facebookresearch/moco). However, this part can be done by actually any unsupervised learning tool.

   TODO
2. Train the whole model supervisedly.  
   `python3 train.py -d='./splits/<YOUR_DATASET>.json' -n='<EXP_NAME>'`

## Testing
`python3 test.py -d='./splits/<YOUR_DATASET>.json' -c='<CHECKPOINT_DIR>'`

# Trained Models

TODO

# Datasets

We use three atomic action datasets.
1. [HAA](https://www.cse.ust.hk/haa/index.html)
2. [Finegym](https://sdolivia.github.io/FineGym/)
3. [MIT](http://moments.csail.mit.edu/)  

Dataset splits and json files can be found under `<REPO_DIR>/splits`, see example dataset jsons or use the scripts there to generate your own. If you want to use other datasets, make sure it has a `<DATASET>/<SPLIT>/<CLASS>/<VIDEO>/<FRAME>` structure.

# Acknowledge

This repo makes use of some great work. Our appreciation for

1. [locuslab / TCN](https://github.com/locuslab/TCN)
2. [fujenchu / relationNet](https://github.com/dragen1860/LearningToCompare-Pytorch)
3. [facebookresearch / moco](https://github.com/facebookresearch/moco)
4. [parlance / ctcdecode](https://github.com/parlance/ctcdecode)
5. [okankop / vidaug](https://github.com/okankop/vidaug)

# Reference

Please refer this paper as

```
@Article{fsaa,
  author  = {Sizhe Song, Xiaoyuan Ni, Yu-Wing Tai, Chi-Keung Tang},
  title   = {Semi-supervised Few-shot Atomic Action Recognition},
  journal = {arXiv preprint arXiv:2011.08410},
  year    = {2020},
}
```
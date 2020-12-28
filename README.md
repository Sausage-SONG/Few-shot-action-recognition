# Semi-supervised Few-shot Atomic Action Recognition

This repo contains the codes for our paper "Semi-supervised Few-shot Atomic Action Recognition". Please check our [paper](https://arxiv.org/abs/2011.08410) and [project page](https://sausage-song.github.io/home/FSAA/) for more details.

<<<<<<< HEAD
![FSAA Architecture](https://github.com/Sausage-SONG/Few-shot-action-recognition/raw/master/FSAA.jpg)
=======
![FSAA Architecture]("FSAA.jpg")
>>>>>>> 9600d47f2f5e1e3e0932d09541f121528ccd979e

Our learning strategies are divided into two parts: 1) train an encoder with unsupervised learning; 2) train the action classification module with supervised learning. Regarding the encoder our model provides fine-grained spatial and temporal video processing with high length flexibility, which embeds the video feature and temporally combines the features with TCN. In terms of classification module, our models provides attention pooling and compares the multi-head relation. Finally, the CTC and MSE loss enables our model for time-invariant few shot classification training.

# Requirements

pytorch >= 1.5.0  
torchvision >= 0.6.0  
numpy >= 1.18.1  
scipy >= 1.4.1  

# Usage

## Installation

1. Clone the repo
2. Install [required packages](#requirements)
3. Download [trained models](#trained-models) to `<REPO_DIR>/models` (Optional)
<<<<<<< HEAD
4. Download the [datasets](#datasets) (Optional)
=======
>>>>>>> 9600d47f2f5e1e3e0932d09541f121528ccd979e

## Training

As mentioned in the [intro](#semi-supervised-few-shot-atomic-action-recognition), our model training has two parts.

1. Train the encoder unsupervisedly.  
   Here we use [MoCo](https://github.com/facebookresearch/moco). However, this part can be done by actually any unsupervised learning tool.

   TODO
2. Train the whole model supervisedly.  
   First modify the settings part in `config.py` then  
   `python3 train.py`

## Testing

1. Modify the settings part of `test.py`
2. `python3 test.py`

# Trained Models

TODO

<<<<<<< HEAD
# Datasets

TODO

=======
>>>>>>> 9600d47f2f5e1e3e0932d09541f121528ccd979e
# Acknowledge

This repo makes use of some great work. Our appreciation for

1. [locuslab / TCN](https://github.com/locuslab/TCN)
2. [facebookresearch / moco](https://github.com/facebookresearch/moco)
3. [parlance / ctcdecode](https://github.com/parlance/ctcdecode)

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
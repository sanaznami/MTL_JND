<h1 align="center"> LASeR-JND: Lightweight Multitask Learning for Robust JND Prediction using Latent Space and Reconstructed Frames</h1>

## Introduction

This is the implementation of LASeR-JND: Lightweight Multitask Learning for Robust JND Prediction using Latent Space and Reconstructed Frames paper in Tensorflow.

**Abstract**

The Just Noticeable Difference (JND) refers to the smallest distortion in an image or video that can be perceived by Human Visual System (HVS), and is widely used in optimizing image/video compression. However, accurate JND modeling is very challenging due to its content dependence, and the complex nature of the HVS. Recent solutions train deep learning based JND prediction models, mainly based on a Quantization Parameter (QP) value, representing a single JND level, and train separate models to predict each JND level. We point out that a single QP-distance is insufficient to properly train a network with millions of parameters, for a complex content-dependent task. Inspired by recent advances in learned compression and multitask learning, we propose to address this problem by (1) learning to reconstruct the JND-quality frames, jointly with the QP prediction, and (2) jointly learning several JND levels to augment the learning performance. We propose a novel solution where first, an effective feature backbone is trained by learning to reconstruct JND-quality frames from the raw frames. Second, JND prediction models are trained based on features extracted from latent space (i.e., compressed domain), or reconstructed JND-quality frames. Third, a multi-JND model is designed, which jointly learns three JND levels, further reducing the prediction error. Extensive experimental results demonstrate that our multi-JND method outperforms the state-of-the-art and achieves an average JND1 prediction error of only 1.57 in QP, and 0.72 dB in PSNR. Moreover, the multitask learning approach, and compressed domain prediction facilitate light-weight inference by significantly reducing the complexity and the number of parameters. 


**The proposed framework**
<p align="center">
  <img src="https://github.com/sanaznami/LASeR-JND/assets/59918141/8edce9b1-a6a6-440c-a468-374375cb7cb8">
</p>

<p align="center">Schematic illustration of previous and proposed approaches: (a) existing approach, (b) the proposed Latent-based JND prediction methods, LAT and E2E-LAT, (c) the proposed Reconstructed-based JND prediction methods, REC and E2E-REC, (d) the proposed Multi-JND (MJ) learning using Latent space, MJ-LAT, and (e) the proposed MJ learning using reconstructed JND-quality frames, MJ-REC.</p>


## Requirements

- Tensorflow
- FFmpeg


## Dataset

Our evaluation is conducted on [VideoSet](https://ieee-dataport.org/documents/videoset) and [MCL-JCI](https://mcl.usc.edu/mcl-jci-dataset/) datasets.


## Pre-trained Models
Our pre-trained models can be downloaded using this link.


## Usage

### Testing

### Training


## Citation

If our work is useful for your research, please cite our paper:

    @inproceedings{nami2024laser,
    	title={LASeR-JND: Lightweight Multitask Learning for Robust JND Prediction using Latent Space and Reconstructed Frames},
	author={Nami, Sanaz and Pakdaman, Farhad and Hashemi, Mahmoud Reza and Shirmohammadi, Shervin and Gabbouj, Moncef},
	journal={},
	year={2024}
    }


## Contact

If you have any question, leave a message here or contact Sanaz Nami (snami@ut.ac.ir, sanaz.nami@tuni.fi).



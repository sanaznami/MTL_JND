<h1 align="center"> Lightweight Multitask Learning for Robust JND Prediction using Latent Space and Reconstructed Frames</h1>

## Introduction

This is the implementation of [Lightweight Multitask Learning for Robust JND Prediction using Latent Space and Reconstructed Frames](https://ieeexplore.ieee.org/document/10500870) paper in Tensorflow.

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
Our pre-trained models can be downloaded using [this link](https://zenodo.org/records/11080876/files/FALCON_IEEETCSVT2024_WP3_v1.0.zip?download=1), from the [Zenodo repository](https://zenodo.org/records/11080876).


## Usage
Our pretrained models are capable of predicting JND values, and they can also be employed for training on a custom dataset.
##### Note: The dataset used for training and testing should have such a structure.

    - rootdir/
         - train/
             - img#1
             - ...
             - JND-Levels.txt (a file containing the 3 JND levels per image: first column for the first JND, second column for the second JND, and third column for the third JND level)
         - valid/
             - img#1
             - ...
             - JND-Levels.txt (a file containing the 3 JND levels per image: first column for the first JND, second column for the second JND, and third column for the third JND level)
         - test/
             - img#1
             - ...
         - jnd1train/
             - img#1
             - ...
         - jnd1valid/
             - img#1
             - ...
         - jnd2train/
             - img#1
             - ...
         - jnd2valid/
             - img#1
             - ...
         - jnd3train/
             - img#1
             - ...
         - jnd3valid/
             - img#1
             - ...
	     
### Testing

For prediction with LAT or REC model, the following commands can be used.

    python3 [LAT.py or REC.py] test --jnd_value [JND1 or JND2 or JND3] --data_dir "Path-to-the-rootdir/" --model_weights_path "Path-to-the-pretrained-model/" --result_path "Path-to-save-test-results/" --JND_Recon_Models_Path "Path-to-the-pretrained-JND-Reconstruction-models/"

For prediction with E2E-LAT or E2E-REC model, the following commands can be used.

    python3 [E2ELAT.py or E2EREC.py] test --jnd_value [JND1 or JND2 or JND3] --data_dir "Path-to-the-rootdir/" --model_weights_path "Path-to-the-pretrained-model/" --result_path "Path-to-save-test-results/" --ImgReconstrution_Model_Path "Path-to-the-pretrained-Img-Reconstruction-models/"

For prediction with MJ-LAT or MJ-REC model, the following commands can be used.

    python3 [MJLAT.py or MJREC.py] test --data_dir "Path-to-the-rootdir/" --model_weights_path "Path-to-the-pretrained-model/" --result_path "Path-to-save-test-results/" --JND_Recon_Models_Path "Path-to-the-pretrained-JND-Reconstruction-models/"


### Training

For training with LAT or REC model, the following commands can be used.

    python3 [LAT.py or REC.py] train --jnd_value [JND1 or JND2 or JND3] --data_dir "Path-to-the-rootdir/" --checkpoint_path "Path-to-save-checkpoints-during-training/" --csv_log_path "Path-to-save-CSV-logs-during-training/" --JND_Recon_Models_Path "Path-to-the-pretrained-JND-Reconstruction-models/" --epochs Number-of-training-epochs --batch_size Batch-size-for-training --learning_rate Learning-rate-for-optimizer

For training with E2E-LAT or E2E-REC model, the following commands can be used.

    python3 [E2ELAT.py or E2EREC.py] train --jnd_value [JND1 or JND2 or JND3] --data_dir "Path-to-the-rootdir/" --checkpoint_path "Path-to-save-checkpoints-during-training/" --csv_log_path "Path-to-save-CSV-logs-during-training/" --ImgReconstrution_Model_Path "Path-to-the-pretrained-Img-Reconstruction-models/" --epochs Number-of-training-epochs --batch_size Batch-size-for-training --learning_rate Learning-rate-for-optimizer

For training with MJ-LAT or MJ-REC model, the following commands can be used.

    python3 [MJLAT.py or MJREC.py] train --jnd_value [JND1 or JND2 or JND3] --data_dir "Path-to-the-rootdir/" --checkpoint_path "Path-to-save-checkpoints-during-training/" --csv_log_path "Path-to-save-CSV-logs-during-training/" --JND_Recon_Models_Path "Path-to-the-pretrained-JND-Reconstruction-models/" --epochs Number-of-training-epochs --batch_size Batch-size-for-training --learning_rate Learning-rate-for-optimizer


## Citation

The attention layer used in LAT-based models is derived from the [Squeeze-and-Excitation Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html) paper.


    @inproceedings{hu2018squeeze,
    	title={Squeeze-and-excitation networks},
	author={Hu, Jie and Shen, Li and Sun, Gang},
	booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
	year={2018}
    }


If our work is useful for your research, please cite our paper:

    @article{nami2024lightweight,
    	title={Lightweight Multitask Learning for Robust JND Prediction using Latent Space and Reconstructed Frames},
	author={Nami, Sanaz and Pakdaman, Farhad and Hashemi, Mahmoud Reza and Shirmohammadi, Shervin and Gabbouj, Moncef},
	journal={IEEE Transactions on Circuits and Systems for Video Technology},
	year={2024},
 	publisher={IEEE}
    }


## Contact

If you have any question, leave a message here or contact Sanaz Nami (snami@ut.ac.ir, sanaz.nami@tuni.fi).



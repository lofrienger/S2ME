# S<sup>2</sup>ME

## Introduction

**S<sup>2</sup>ME: Spatial-Spectral Mutual Teaching and Ensemble Learning for Scribble-supervised Polyp Segmentation**

An Wang, Mengya Xu, Yang Zhang, Mobarakol Islam, and Hongliang Ren

Medical Image Computing and Computer-Assisted Intervention (MICCAI) - 2023 

*Early Accepted (Top 14% of 2253 manuscripts)*

To our best knowledge, we propose the first spatial-spectral dual-branch network structure for weakly-supervised medical image segmentation that efficiently leverages cross-domain patterns with collaborative mutual teaching and ensemble learning. Our pixel-level entropy-guided fusion strategy advances the reliability of the aggregated pseudo labels, which provides valuable supplementary supervision signals. Moreover, we optimize the segmentation model with the hybrid mode of loss supervision from scribbles and pseudo labels in a holistic manner and witness improved outcomes. With extensive in-domain and out-ofdomain evaluation on four public datasets, our method shows superior accuracy, generalization, and robustness, indicating its clinical significance in alleviating data-related issues such as data shift and corruption which are commonly encountered in the medical field. 

![s2me](Image/s2me.png?raw=true "s2me")

## Environment
- NVIDIA RTX3090
- Python 3.8
- Pytorch 1.10
- Check [environment.yml](code/environment.yml) for more dependencies.

## Usage
1. Dataset
    - SUN-SEG: Download from [SUN-SEG](https://github.com/GewelsJI/VPS), then follow the json files in the folder _data/polyp_ for splits. 
    - Kvasir-SEG: Download from [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/).
    - CVC-ClinicDB: Download from [CVC-ClinicDB](https://www.kaggle.com/datasets/balraj98/cvcclinicdb).
    - PolypGen: Download from [PolypGen](https://www.synapse.org/#!Synapse:syn26376615/wiki/613312).

2. Training and Testing
 
- Command:
  
    ```
    CUDA_VISIBLE_DEVICES=1 python train_s2me.py --model1 unet --model2 ynet_ffc --sup Scribble --exp s2me-ent-css_5.0_25k --mps True --mps_type entropy --cps True
    ```
- Some essential hyperparameters:
  
    - *mps* and *mps_type*: Whether apply **Mutual Teaching** and its type (*entropy* is our entropy-guided fusion);
    - *cps*: Apply **Ensemble Learning** or not;
    - Refer to [train_s2me.py](code/train_s2me.py) for more explanation on other hyperparameters.

- Trained model and training log
    - One trained model which yields our best result on the SUN-SEG dataset is available in the folder *model*.


1. Test Result

    - In-domain quantitative performance
    <div align=center>
    <img src=Image/Table1.png width=600 height=250>
    <div align=left>

    - In-domain qualitative performance
    <div align=center>
    <img src=Image/Fig2.png width=500 height=140>
    <div align=left>

    - Generalization performance
    <div align=center>
    <img src=Image/TableGen1.png width=450 height=190>
    <div align=left>

    <div align=center>
    <img src=Image/TableGen2.png width=450 height=190>
    <div align=left>

    <div align=center>
    <img src=Image/TableGen3.png width=450 height=190>
    <div align=left>

    - Ablation Studies
    <div align=center>
    <img src=Image/Table3.png width=550 height=110>
    <div align=left>

    <div align=center>
    <img src=Image/Table45.png width=550 height=140>
    <div align=left>

## Acknowledgement
Some of the codes are borrowed/refer from below repositories:
- [WSL4MIS](https://github.com/HiLab-git/WSL4MIS)
- [PyMIC](https://github.com/HiLab-git/PyMIC)
- [TGANet](https://github.com/nikhilroxtomar/TGANet)

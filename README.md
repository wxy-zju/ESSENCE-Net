# ESSENCE-Net
## TPAMI-2025 [Revisiting Supervised Learning-based Photometric Stereo Networks](https://ieeexplore.ieee.org/document/10948383)
Xiaoyao Wei, Zongrui Li, Binjie Ding, Boxin Shi, Xudong Jiang, Gang Pan, Yanlong Cao, and Qian Zheng
https://ieeexplore.ieee.org/document/10948383

## Dependencies
ESSENCE-Net is implemented in PyTorch with Ubuntu 18.04 and an NVIDIA GeForce RTX 3090 GPU (24GB).
* Python 3.8.5
* PyTorch (version = 1.90)
* numpy
* scipy
* CUDA-11.1
* einops
## Testing 
### Test on the DiLiGenT dataset

* Dense setups (96 input images)
```
python main_test.py --in_img_num 96
```
* Sparse setups (10 input images)
```
python main_test.py --in_img_num 10
```
### Test on the DiLiGenT10<sup>2</sup> dataset, DiLiGenT-Π dataset, and DiLiGenRT dataset
The ground truth of the DiLiGenT10<sup>2</sup> dataset, DiLiGenT-Π dataset, and DiLiGenRT dataset is not publicly available. You can use these codes to estimate normal maps and submit the estimated normal maps to the corresponding website for evaluation of normal errors.

## Training
The training code will be made available soon.

## Results on the DiLiGenT dataset, DiLiGenT10<sup>2</sup> dataset, DiLiGenT-Π dataset, and DiLiGenRT dataset
We have provided the estimated surface normal maps (error maps) on the DiLiGenT, DiLiGenT10<sup>2</sup>, DiLiGenT-Π, and DiLiGenRT benchmark datasets under 96/100 input images in ./pre_trained_model/.

## Acknowledgement
Our code is partially based on https://github.com/guanyingc/PS-FCN.

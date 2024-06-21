# SGHPS-Net
## Revisiting Deep Features in Supervised Learning-based Photometric Stereo Networks
Xiaoyao Wei, Qian Zheng, Binjie Ding, Zongrui Li, Gang Pan, Xudong Jiang, Boxin Shi, and Yanlong Cao

## Dependencies
SGHPS-Net is implemented in PyTorch with Ubuntu 18.04 and an NVIDIA GeForce RTX 3090 GPU (24GB).
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
### Test on the DiLiGenT10<sup>2</sup> dataset and DiLiGenT-Π dataset
The ground truth of the the DiLiGenT10<sup>2</sup> dataset and DiLiGenT-Π dataset is not open, you can use these codes to estimate normal maps and submit the estimated normal maps to the corresponding website for evaluation of normal errors.

## Training
The training code will be made available soon.

## Results on the DiLiGenT benchmark dataset
We have provided the estimated surface normal maps and error maps on the DiLiGenT benchmark dataset under 96 input images in ./pre_trained_model/test/.

## Acknowledgement
Our code is partially based on https://github.com/guanyingc/PS-FCN.

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
### Test on the DiLiGenT dataset

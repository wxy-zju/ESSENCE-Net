import torch
import numpy as np


def arrayToTensor(array):
    if array is None:
        return array
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array)
    return tensor.float()

def normalize_max(imgs):
    h, w, c = imgs[0].shape
    imgs = [img.reshape(-1, 1) for img in imgs]
    img = np.hstack(imgs)
    max = img.max(1)
    img = img / (max.reshape(-1,1) + 1e-10)
    imgs = np.split(img, img.shape[1], axis=1)
    imgs = [img.reshape(h, w, -1) for img in imgs]
    return imgs

def cal_shading(lights,normal,img):
    shading = np.zeros([img.shape[0],img.shape[1],img.shape[2]//3])
    for i in range(lights.shape[0]):
        npl = lights[i,:].reshape(1,1,-1)*normal
        npl = npl.sum(2)
        shading[:,:,i] = npl
    return shading


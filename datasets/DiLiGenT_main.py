from __future__ import division
import os
import random
import numpy as np
from imageio import imread
import scipy.io as sio
import torch
import torch.utils.data as data
from datasets import pms_transforms
from . import util

class DiLiGenT_main(data.Dataset):
    def __init__(self, args, split='train'):
        self.root   = os.path.join(args.bm_dir)
        self.args   = args
        self.objs   = util.readList(os.path.join(self.root, 'objects.txt'), sort=False)
        if self.args.in_img_num!=96:
            self.objs=self.objs*100 #sparse
        self.names  = util.readList(os.path.join(self.root, 'filenames.txt'), sort=False)
        self.l_dir  = util.light_source_directions()
        print('[%s Data] \t%d objs %d lights. Root: %s' % 
                (split, len(self.objs), len(self.names), self.root))
        self.intens = {}
        intens_name = 'light_intensities.txt'
        print('Files for intensity: %s' % (intens_name))
        for obj in self.objs:
            self.intens[obj] = np.genfromtxt(os.path.join(self.root, obj, intens_name))

    def _getMask(self, obj):
        mask = imread(os.path.join(self.root, obj, 'mask.png'))
        if mask.ndim > 2: mask = mask[:,:,0]
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        return mask / 255.0

    def __getitem__(self, index):
        np.random.seed(index)
        obj = self.objs[index]
        np.random.seed(index//10)   # for sparse
        select_idx = np.random.permutation(len(self.names))[:self.args.in_img_num]

        if obj == 'bearPNG':
            select_idx=random.sample(range(20, 96), min(self.args.in_img_num,76))
        img_list   = [os.path.join(self.root, obj, self.names[i]) for i in select_idx]
        intens     = [np.diag(1 / self.intens[obj][i]) for i in select_idx]
        normal_path = os.path.join(self.root, obj, 'Normal_gt.mat')
        normal = sio.loadmat(normal_path)
        normal = normal['Normal_gt']

        imgs = []
        for idx, img_name in enumerate(img_list):
            img = imread(img_name).astype(np.float32) / 255.0
            img = np.dot(img, intens[idx])
            img=img.clip(0,1)
            imgs.append(img)
        if self.args.normalize:
            imgs = pms_transforms.normalize_max(imgs)
        img = np.concatenate(imgs, 2)

        mask = self._getMask(obj)
        img  = img * mask.repeat(img.shape[2], 2)
        lights = self.l_dir[select_idx]
        shading_gt = pms_transforms.cal_shading(lights, normal, img)
        shading_gt = shading_gt * mask

        item = {'N': normal, 'img': img, 'mask': mask, 'shading_gt': shading_gt}
        for k in item.keys(): 
            item[k] = pms_transforms.arrayToTensor(item[k])
        item['light'] = torch.from_numpy(self.l_dir[select_idx]).view(-1, 1, 1).float()
        item['obj'] = obj
        return item

    def __len__(self):
        return len(self.objs)

import datetime, time
import os
import numpy as np
import torch
import torchvision.utils as vutils

def dictToString(dicts, start='\t', end='\n'):
    strs = ''
    for k, v in sorted(dicts.items()):
        strs += '%s%s: %s%s' % (start, str(k), str(v), end)
    return strs

class Logger(object):
    def __init__(self, args):
        self.times = {'init': time.time()}
        self.args = args
        self.printArgs()

    def printArgs(self):
        strs = '------------ Options -------------\n'
        strs += '{}'.format(dictToString(vars(self.args)))
        strs += '-------------- End ----------------\n'
        print(strs)

    def getTimeInfo(self):
        time_elapsed = (time.time() - self.times['init']) / 3600.0

        return time_elapsed,

    def printItersSummary(self, opt):
        epoch, iters, batch = opt['epoch'], opt['iters'], opt['batch']
        strs = ' | {}'.format(str.upper(opt['split']))
        strs += ' Iter [{}/{}] '.format(iters, batch)
        if 'timer' in opt.keys():
            print(opt['timer'].timeToString())
        if 'recorder' in opt.keys():
            print(opt['recorder'].iterRecToString(opt['split'], epoch))

    def printEpochSummary(self, opt):
        split = opt['split']
        epoch = opt['epoch']
        print('---------- {} Epoch {} Summary -----------'.format(str.upper(split), epoch))
        print(opt['recorder'].epochRecToString(split, epoch))



    def saveSplit(self, res, save_prefix):
        n, c, h, w = res.shape
        name_list=['_normal_gt','_normal_est','_error_map']
        for i in range(n):
            vutils.save_image(res[i], save_prefix + name_list[i]+'.png')
    def saveImgResults(self, results, split, obj):
        print(obj)
        res = [img.cpu() for img in results]
        res=torch.cat(res)

        save_dir = os.path.join(os.path.dirname(self.args.retrain), split)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_prefix_ = os.path.join(save_dir, obj)
        self.saveSplit(res, save_prefix_)
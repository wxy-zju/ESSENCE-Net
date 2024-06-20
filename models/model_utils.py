import os
import torch
import torch.nn as nn

def getInput(data):
    input_list = [data['input']]
    input_list.append(data['l'])
    return input_list

def parseData(args, sample, timer=None):
    input, target, mask,obj = sample['img'], sample['N'], sample['mask'],sample['obj']
    shading_gt = sample['shading_gt']
    if timer: timer.updateTime('ToCPU')
    if args.cuda:
        input  = input.cuda(); target = target.cuda(); mask = mask.cuda(); shading_gt = shading_gt.cuda();

    input_var  = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)
    mask_var   = torch.autograd.Variable(mask, requires_grad=False);
    shading_gt_var = torch.autograd.Variable(shading_gt)

    if timer: timer.updateTime('ToGPU')
    data = {'input': input_var, 'tar': target_var, 'm': mask_var, 'shading_gt':shading_gt_var,'obj':obj}
    light = sample['light'].expand_as(input)
    if args.cuda: light = light.cuda()
    light_var = torch.autograd.Variable(light);
    data['l'] = light_var

    return data 



def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def loadCheckpoint(path, model, cuda=True):
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

def deconv(cin, cout):
    return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
            )
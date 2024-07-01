import os
import sys
import torch
sys.path.append('.')

import test_utils

from datasets import custom_data_loader
from options  import base_opts
from models import custom_model
from utils  import logger, recorders

args = base_opts.BaseOpts().parse()
log  = logger.Logger(args)
os.environ['CUDA_VISIBLE_DEVICES']='0'
if args.in_img_num==10:
    args.retrain='./pre_trained_model/ESSENCENet_sparse.tar'
def main(args):
    test_loader = custom_data_loader.benchmarkLoader(args)
    model    = custom_model.buildModel(args)
    recorder = recorders.Records()
    test_utils.test(args, 'test', test_loader, model, log, 1, recorder)

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)

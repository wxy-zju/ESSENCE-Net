import argparse

class BaseOpts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()
    def initialize(self):
        #### Device Arguments ####
        self.parser.add_argument('--cuda',        default=True,  action='store_false')
        self.parser.add_argument('--time_sync',   default=False, action='store_true')
        self.parser.add_argument('--workers',     default=28,     type=int)
        self.parser.add_argument('--seed',        default=0,     type=int)

        #### Model Arguments ####
        self.parser.add_argument('--normalize',  default=True, action='store_false')
        self.parser.add_argument('--in_img_num', default=96,    type=int)
        self.parser.add_argument('--retrain',    default='./pre_trained_model/SGHPSNet_dense.tar')
        self.parser.add_argument('--benchmark', default='DiLiGenT_main')
        self.parser.add_argument('--bm_dir', default='./DiLiGenT/pmsData_crop')
        self.parser.add_argument('--model', default='ESSENCE_Net')
        self.parser.add_argument('--test_batch', default=1, type=int)

    def parse(self):
            self.args = self.parser.parse_args()
            return self.args

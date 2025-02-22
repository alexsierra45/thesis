import os
import torch

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] =str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def infer(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

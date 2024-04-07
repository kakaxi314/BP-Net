# -*- coding: utf-8 -*-
# @File : schedulers.py
# @Project: BP-Net
# @Author : jie
# @Time : 5/11/22 3:50 PM
import random
import sys
from torch.optim.lr_scheduler import StepLR, MultiStepLR, OneCycleLR, LambdaLR, LinearLR, ExponentialLR
import numpy as np
import torch


def NoiseLR(**kwargs):
    lr_sched = getattr(sys.modules[__name__], kwargs.pop('lr_sched', 'OneCycleLR'))

    class sched(lr_sched):
        def __init__(self, **kwargs):
            self.noise_pct = kwargs.pop('noise_pct', 0.1)
            self.noise_seed = kwargs.pop('noise_seed', 0)
            super().__init__(**kwargs)

        def get_lr(self):
            """
            lrn: Learning Rate with Noise
            """
            g = torch.Generator()
            g.manual_seed(self.noise_seed + self.last_epoch)
            noise = 2 * torch.rand(1, generator=g).item() - 1
            lrs = super().get_lr()
            lrn = []
            for lr in lrs:
                lrn.append(lr * (1 + self.noise_pct * noise))
            return lrn

    return sched(**kwargs)


#!/usr/bin/env
# -*- coding: utf-8 -*-
# @Filename : utils
# @Date : 2022-05-06
# @Project: BP-Net
# @AUTHOR : jie

import os
import torch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf
import cv2
import augs
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from collections import OrderedDict
import math

__all__ = [
    'AverageMeter',
    'Trainer',
]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
        self.cfg.gpu_id = self.cfg.gpus[self.rank]
        self.init_gpu()
        self.ddp = len(self.cfg.gpus) > 1
        self.iter = 0
        self.epoch = 0
        self.best_metric_ema = 100
        #####################################################################################
        self.log = self.init_log()
        self.init_device()
        self.init_seed()
        self.writer = self.init_viz()
        self.trainloader, self.testloader = self.init_dataset()
        net = self.init_net()
        criterion = self.init_loss()
        metric = self.init_metric()
        self.net, self.criterion, self.metric = self.init_cuda(net, criterion, metric)
        self.net_ema = self.init_ema()
        if self.ddp:
            self.net = DDP(self.net)
        self.optimizer = self.init_optim()
        self.lr_scheduler = self.init_sched_lr()
        self.lr_iter = OmegaConf.select(self.cfg.sched.lr, 'iter', default=False)
        self.clip = self.init_clip()

    def init_log(self):
        return Blank() if self.rank else logging.getLogger(f'{self.cfg.name}')

    def init_device(self):
        torch.cuda.set_device(f'cuda:{self.cfg.gpu_id}')
        if self.ddp:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
        self.ddp_log(f'device is {self.cfg.gpu_id}', always=True)

    def init_seed(self):
        manual_seed = self.cfg.manual_seed
        self.ddp_log(f"Random Seed: {manual_seed:04d}")
        torch.initial_seed()
        random.seed(manual_seed)
        np.random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

    def init_viz(self):
        if self.rank:
            return Blank()
        else:
            writer_name = os.path.join('runs', f'{self.cfg.name}')
            return SummaryWriter(writer_name)

    def init_dataset(self):
        trainset = instantiate(self.cfg.data.trainset)
        testset = instantiate(self.cfg.data.testset)
        if self.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset,
                num_replicas=len(self.cfg.gpus),
                rank=self.rank,
                shuffle=True,
            )
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                testset,
                num_replicas=len(self.cfg.gpus),
                rank=self.rank,
                shuffle=False,
            )
        else:
            train_sampler = None
            test_sampler = None
        self.train_sampler = train_sampler
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.cfg.train_batch_size,
                                                  num_workers=self.cfg.num_workers, shuffle=(train_sampler is None),
                                                  sampler=train_sampler,
                                                  drop_last=True, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.cfg.test_batch_size,
                                                 num_workers=self.cfg.num_workers, shuffle=False,
                                                 sampler=test_sampler,
                                                 drop_last=True, pin_memory=True)
        self.ddp_log(f'num_train = {len(trainloader)}, num_test = {len(testloader)}')
        return trainloader, testloader


    def init_net(self):
        model = instantiate(self.cfg.net.model)
        if 'chpt' in self.cfg:
            self.ddp_log(f'resume CHECKPOINTS')
            save_path = os.path.join('checkpoints', self.cfg.chpt)
            cp = torch.load(os.path.join(save_path, 'result_ema.pth'), map_location=torch.device('cpu'))
            model.load_state_dict(cp['net'], strict=True)
            self.best_metric_ema = cp['best_metric_ema']
            del cp
        if self.ddp and OmegaConf.select(self.cfg.net, 'sbn', default=False):
            self.ddp_log('sbn')
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            """
            SBN is not compatible with torch.compile
            """
            if OmegaConf.select(self.cfg.net, 'compile', default=False):
                self.ddp_log('compile')
                model = torch.compile(model)
        return model

    def init_ema(self):
        return instantiate(self.cfg.net.ema, model=self.net, ddp=self.ddp)

    def init_loss(self):
        return instantiate(self.cfg.loss)

    def init_metric(self):
        return instantiate(self.cfg.metric)

    def init_cuda(self, *modules):
        modules = [module.to(f'cuda:{self.cfg.gpu_id}') for module in modules]
        return modules

    def init_gpu(self):
        with torch.cuda.device(f'cuda:{self.cfg.gpu_id}'):
            torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    def init_optim(self):
        optim = instantiate(self.cfg.optim, _partial_=True)
        return optim(params=config_param(self.net))

    def init_clip(self):
        if 'clip' in self.cfg.net:
            return instantiate(self.cfg.net.clip, _partial_=True)
        else:
            return None

    def init_sched_lr(self):
        sched = instantiate(self.cfg.sched.lr.policy, _partial_=True)
        return sched(optimizer=self.optimizer)


    def ddp_log(self, content, always=False):
        # self.log.info(f'{content}')
        if (not self.rank) or always:
            self.log.info(f'{content}')

    def ddp_cout(self, content, always=False):
        # tqdm.write(f'{content}')
        if (not self.rank) or always:
            tqdm.write(f'{content}')

    def save_state(self):
        if self.rank:
            return
        save_path = os.path.join('checkpoints', self.cfg.name)
        os.makedirs(save_path, exist_ok=True)
        model = self.net_ema.module
        if hasattr(model, 'module'):
            model = model.module
        model_state_dict = model.state_dict()
        state_dict = {
            'net': model_state_dict,
            'epoch': self.epoch,
            'best_metric_ema': self.best_metric_ema,
        }
        torch.save(state_dict, os.path.join(save_path, 'result_ema.pth'))

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.writer.close()
        self.ddp_log(f'best_metric_ema={self.best_metric_ema:.4f}')


def config_param(model):
    param_groups = []
    other_params = []
    for name, param in model.named_parameters():
        if len(param.shape) == 1:
            g = {'params': [param], 'weight_decay': 0.0}
            param_groups.append(g)
        else:
            other_params.append(param)
    param_groups.append({'params': other_params})
    return param_groups


def set_requires_grad(model, requires_grad=True):
    for p in model.parameters():
        p.requires_grad = requires_grad



class Blank(object):
    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            return None
        return wrapper

FloorDiv = lambda a, b: a // b

CeilDiv = lambda a, b: math.ceil(a / b)

Div = lambda a, b: a / b

Mul = lambda a, b: a * b



# -*- coding: utf-8 -*-
# @File : train_amp.py
# @Project: BP-Net
# @Author : jie
# @Time : 10/27/21 3:58 PM

import torch
from tqdm import tqdm
import hydra
from PIL import Image
import os
from omegaconf import OmegaConf
from utils import *


def test(run, mode='selval', save=False):
    dataloader = run.testloader
    net = run.net_ema.module
    net.eval()
    tops = [AverageMeter() for i in range(len(run.metric.metric_name))]
    if save:
        dir_path = f'results/{run.cfg.name}/{mode}'
        os.makedirs(dir_path, exist_ok=True)
    with torch.no_grad():
        for idx, datas in enumerate(
                tqdm(dataloader, desc="test ", dynamic_ncols=True, leave=False, disable=run.rank)):
            datas = run.init_cuda(*datas)
            output = net(*datas[:-1])
            if isinstance(output, (list, tuple)):
                output = output[-1]
            precs = run.metric(output, datas[-1])
            for prec, top in zip(precs, tops):
                top.update(prec.mean().detach().cpu().item())
            if save:
                for i in range(output.shape[0]):
                    index = idx * output.shape[0] + i
                    file_path = os.path.join(dir_path, f'{index:010d}.png')
                    img = (output[i, 0] * 256.0).detach().cpu().numpy().astype('uint16')
                    Img = Image.fromarray(img)
                    Img.save(file_path)
    logs = ""
    for name, top in zip(run.metric.metric_name, tops):
        logs += f" {name}:{top.avg:.7f} "
    run.ddp_log(logs, always=True)


@hydra.main(config_path='configs', config_name='config', version_base='1.2')
def main(cfg):
    with Trainer(cfg) as run:
        test(run, mode=cfg.data.testset.mode, save=OmegaConf.select(cfg, 'save', default=False))



if __name__ == '__main__':
    main()

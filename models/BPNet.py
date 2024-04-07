# -*- coding: utf-8 -*-
# @File : BPNet.py
# @Project: BP-Net
# @Author : jie
# @Time : 4/8/23 12:43 PM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from .utils import Conv1x1, Basic2d, BasicBlock, weights_init, inplace_relu, GenKernel, PMP

__all__ = [
    'Pre_MF_Post',
]


class Net(nn.Module):
    """
    network
    """

    def __init__(self, block=BasicBlock, bc=16, img_layers=[2, 2, 2, 2, 2, 2],
                 drop_path=0.1, norm_layer=nn.BatchNorm2d, padding_mode='zeros', drift=1e6):
        super().__init__()
        self.drift = drift
        self._norm_layer = norm_layer
        self._padding_mode = padding_mode
        self._img_dpc = 0
        self._img_dprs = np.linspace(0, drop_path, sum(img_layers))

        self.inplanes = bc * 2
        self.conv_img = nn.Sequential(
            Basic2d(3, bc * 2, norm_layer=norm_layer, kernel_size=3, padding=1),
            self._make_layer(block, bc * 2, 2, stride=1)
        )

        self.layer1_img = self._make_layer(block, bc * 4, img_layers[1], stride=2)

        self.inplanes = bc * 4
        self.layer2_img = self._make_layer(block, bc * 8, img_layers[2], stride=2)

        self.inplanes = bc * 8
        self.layer3_img = self._make_layer(block, bc * 16, img_layers[3], stride=2)

        self.inplanes = bc * 16
        self.layer4_img = self._make_layer(block, bc * 16, img_layers[4], stride=2)

        self.inplanes = bc * 16
        self.layer5_img = self._make_layer(block, bc * 16, img_layers[5], stride=2)

        self.pred0 = PMP(level=0, in_ch=bc * 4, out_ch=bc * 2, drop_path=drop_path, pool=False)
        self.pred1 = PMP(level=1, in_ch=bc * 8, out_ch=bc * 4, drop_path=drop_path)
        self.pred2 = PMP(level=2, in_ch=bc * 16, out_ch=bc * 8, drop_path=drop_path)
        self.pred3 = PMP(level=3, in_ch=bc * 16, out_ch=bc * 16, drop_path=drop_path)
        self.pred4 = PMP(level=4, in_ch=bc * 16, out_ch=bc * 16, drop_path=drop_path)
        self.pred5 = PMP(level=5, in_ch=bc * 16, out_ch=bc * 16, drop_path=drop_path, up=False)

    def forward(self, I, S, K):
        """
        I: Bx3xHxW
        S: Bx1xHxW
        K: Bx3x3
        """
        output = []
        XI0 = self.conv_img(I)
        XI1 = self.layer1_img(XI0)
        XI2 = self.layer2_img(XI1)
        XI3 = self.layer3_img(XI2)
        XI4 = self.layer4_img(XI3)
        XI5 = self.layer5_img(XI4)

        fout, dout = self.pred5(fout=None, dout=None, XI=XI5, S=S, K=K)
        output.append(F.interpolate(dout, scale_factor=2 ** 5, mode='bilinear', align_corners=True))

        fout, dout = self.pred4(fout=fout, dout=dout, XI=XI4, S=S, K=K)
        output.append(F.interpolate(dout, scale_factor=2 ** 4, mode='bilinear', align_corners=True))

        fout, dout = self.pred3(fout=fout, dout=dout, XI=XI3, S=S, K=K)
        output.append(F.interpolate(dout, scale_factor=2 ** 3, mode='bilinear', align_corners=True))

        fout, dout = self.pred2(fout=fout, dout=dout, XI=XI2, S=S, K=K)
        output.append(F.interpolate(dout, scale_factor=2 ** 2, mode='bilinear', align_corners=True))

        fout, dout = self.pred1(fout=fout, dout=dout, XI=XI1, S=S, K=K)
        output.append(F.interpolate(dout, scale_factor=2 ** 1, mode='bilinear', align_corners=True))

        fout, dout = self.pred0(fout=fout, dout=dout, XI=XI0, S=S, K=K)
        output.append(dout)
        return output

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        padding_mode = self._padding_mode
        downsample = None
        if norm_layer is None:
            bias = True
            norm_layer = nn.Identity
        else:
            bias = False
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride, bias=bias),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer, padding_mode=padding_mode,
                  drop_path=self._img_dprs[self._img_dpc]))
        self._img_dpc += 1
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, padding_mode=padding_mode,
                                drop_path=self._img_dprs[self._img_dpc]))
            self._img_dpc += 1

        return nn.Sequential(*layers)


def Pre_MF_Post():
    """
    Pre.+MF.+Post.
    """
    net = Net()
    net.apply(functools.partial(weights_init, mode='trunc'))
    for m in net.modules():
        if isinstance(m, GenKernel) and m.conv[1].conv.bn.weight is not None:
            nn.init.constant_(m.conv[1].conv.bn.weight, 0)
    net.apply(inplace_relu)
    return net

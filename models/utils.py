#!/usr/bin/env
# -*- coding: utf-8 -*-
# @Filename : utils
# @Date : 2022-05-10
# @Project: BP-Net
# @AUTHOR : jie

from copy import deepcopy
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math
import numpy as np
from collections import OrderedDict
import BpOps
import torch.distributed as dist
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath
from torch.cuda.amp import custom_fwd, custom_bwd
import functools

__all__ = [
    'EMA',
    'Conv1x1',
    'Conv3x3',
    'Basic2d',
    'weights_init',
    'inplace_relu',
    'PMP',
]


class BpDist(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, xy, idx, Valid, num, H, W):
        """
        """
        assert xy.is_contiguous()
        assert Valid.is_contiguous()
        _, Cc, M = xy.shape
        B = Valid.shape[0]
        N = H * W
        args = torch.zeros((B, num, N), dtype=torch.long, device=xy.device)
        IPCnum = torch.zeros((B, Cc, num, N), dtype=xy.dtype, device=xy.device)
        for b in range(B):
            Pc = torch.masked_select(xy, Valid[b:b + 1].view(1, 1, N)).reshape(1, 2, -1)
            BpOps.Dist(Pc, IPCnum[b:b + 1], args[b:b + 1], H, W)
            idx_valid = torch.masked_select(idx, Valid[b:b + 1].view(1, 1, N))
            args[b:b + 1] = torch.index_select(idx_valid, 0, args[b:b + 1].reshape(-1)).reshape(1, num, N)
        return IPCnum, args

    @staticmethod
    @custom_bwd
    def backward(ctx, ga=None, gb=None):
        return None, None, None, None


class BpConvLocal(Function):
    @staticmethod
    def forward(ctx, input, weight):
        assert input.is_contiguous()
        assert weight.is_contiguous()
        ctx.save_for_backward(input, weight)
        output = BpOps.Conv2dLocal_F(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input, grad_weight = BpOps.Conv2dLocal_B(input, weight, grad_output)
        return grad_input, grad_weight


bpdist = BpDist.apply
bpconvlocal = BpConvLocal.apply


class EMA(nn.Module):
    """ Model Exponential Moving Average V2 borrow from timm https://timm.fast.ai/

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, ddp=False):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        if ddp:
            self.broadcast()
        self.decay = decay

    def broadcast(self):
        for ema_v in self.module.state_dict().values():
            dist.broadcast(ema_v, src=0, async_op=False)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def weights_init(m, mode='trunc'):
    from torch.nn.init import _calculate_fan_in_and_fan_out
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if hasattr(m, 'weight'):
            if mode == 'trunc':
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight.data)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                torch.nn.init.trunc_normal_(m.weight.data, mean=0, std=std)
            elif mode == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data)
            else:
                raise ValueError(f'unknown mode = {mode}')
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    if classname.find('Conv1d') != -1:
        if hasattr(m, 'weight'):
            if mode == 'trunc':
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight.data)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                torch.nn.init.trunc_normal_(m.weight.data, mean=0, std=std)
            elif mode == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data)
            else:
                raise ValueError(f'unknown mode = {mode}')
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        if mode == 'trunc':
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight.data)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            torch.nn.init.trunc_normal_(m.weight.data, mean=0, std=std)
        elif mode == 'xavier':
            torch.nn.init.xavier_normal_(m.weight.data)
        else:
            raise ValueError(f'unknown mode = {mode}')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


def Conv1x1(in_planes, out_planes, stride=1, bias=False, groups=1, dilation=1, padding_mode='zeros'):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, padding_mode='zeros', bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, padding_mode=padding_mode, groups=groups, bias=bias, dilation=dilation)


class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1, padding_mode='zeros',
                 act=nn.ReLU, stride=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=False, padding_mode=padding_mode)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.conv = nn.Sequential(OrderedDict([('conv', conv)]))
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', act())

    def forward(self, x):
        out = self.conv(x)
        return out



def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, act=nn.ReLU):
        super().__init__()
        if norm_layer is None:
            bias = True
            norm_layer = nn.Identity
        else:
            bias = False
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4,
                                       stride=2, padding=1, bias=bias)
        self.bn = norm_layer(out_channels)
        self.relu = act()

    def forward(self, x):
        out = self.conv(x.contiguous())
        out = self.bn(out)
        out = self.relu(out)
        return out


class UpCC(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, norm_layer=None, kernel_size=3, padding=1,
                 padding_mode='zeros', act=nn.ReLU):
        super().__init__()
        self.upf = Basic2dTrans(in_channels, out_channels, norm_layer=norm_layer, act=act)
        self.conv = Basic2d(mid_channels + out_channels, out_channels,
                            norm_layer=norm_layer, kernel_size=kernel_size,
                            padding=padding, padding_mode=padding_mode, act=act)

    def forward(self, x, y):
        """
        """
        out = self.upf(x)
        fout = torch.cat([out, y], dim=1)
        fout = self.conv(fout)
        return fout

class GenKernel(nn.Module):
    def __init__(self, in_channels, pk, norm_layer=nn.BatchNorm2d, act=nn.ReLU, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.conv = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, pk * pk - 1, norm_layer=norm_layer, act=nn.Identity),
        )

    def forward(self, fout):
        weight = self.conv(fout)
        weight_sum = torch.sum(weight.abs(), dim=1, keepdim=True)
        weight = torch.div(weight, weight_sum + self.eps)
        weight_mid = 1 - torch.sum(weight, dim=1, keepdim=True)
        weight_pre, weight_post = torch.split(weight, [weight.shape[1] // 2, weight.shape[1] // 2], dim=1)
        weight = torch.cat([weight_pre, weight_mid, weight_post], dim=1).contiguous()
        return weight


class CSPN(nn.Module):
    """
    implementation of CSPN++
    """

    def __init__(self, in_channels, pt, norm_layer=nn.BatchNorm2d, act=nn.ReLU, eps=1e-6):
        super().__init__()
        self.pt = pt
        self.weight3x3 = GenKernel(in_channels, 3, norm_layer=norm_layer, act=act, eps=eps)
        self.weight5x5 = GenKernel(in_channels, 5, norm_layer=norm_layer, act=act, eps=eps)
        self.weight7x7 = GenKernel(in_channels, 7, norm_layer=norm_layer, act=act, eps=eps)
        self.convmask = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=nn.Sigmoid),
        )
        self.convck = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=functools.partial(nn.Softmax, dim=1)),
        )
        self.convct = nn.Sequential(
            Basic2d(in_channels + 3, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=functools.partial(nn.Softmax, dim=1)),
        )

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, fout, hn, h0):
        weight3x3 = self.weight3x3(fout)
        weight5x5 = self.weight5x5(fout)
        weight7x7 = self.weight7x7(fout)
        mask3x3, mask5x5, mask7x7 = torch.split(self.convmask(fout) * (h0 > 1e-3).float(), 1, dim=1)
        conf3x3, conf5x5, conf7x7 = torch.split(self.convck(fout), 1, dim=1)
        hn3x3 = hn5x5 = hn7x7 = hn
        hns = [hn, ]
        for i in range(self.pt):
            hn3x3 = (1. - mask3x3) * bpconvlocal(hn3x3, weight3x3) + mask3x3 * h0
            hn5x5 = (1. - mask5x5) * bpconvlocal(hn5x5, weight5x5) + mask5x5 * h0
            hn7x7 = (1. - mask7x7) * bpconvlocal(hn7x7, weight7x7) + mask7x7 * h0
            if i == self.pt // 2 - 1:
                hns.append(conf3x3 * hn3x3 + conf5x5 * hn5x5 + conf7x7 * hn7x7)
        hns.append(conf3x3 * hn3x3 + conf5x5 * hn5x5 + conf7x7 * hn7x7)
        hns = torch.cat(hns, dim=1)
        wt = self.convct(torch.cat([fout, hns], dim=1))
        hn = torch.sum(wt * hns, dim=1, keepdim=True)
        return hn


class Coef(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels=3, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=1, padding=padding, bias=True)

    def forward(self, x):
        feat = self.conv(x)
        XF, XB, XW = torch.split(feat, [1, 1, 1], dim=1)
        return XF, XB, XW


class Dist(nn.Module):
    """
    """

    def __init__(self, num):
        super().__init__()
        """
        """
        self.num = num

    def forward(self, S, xx, yy):
        """
        """
        num = self.num
        B, _, height, width = S.shape
        N = height * width
        S = S.reshape(B, 1, N)
        Valid = (S > 1e-3)
        xy = torch.stack((xx, yy), axis=0).reshape(1, 2, -1).float()
        idx = torch.arange(N, device=S.device).reshape(1, 1, N)
        Ofnum, args = bpdist(xy, idx, Valid, num, height, width)
        return Ofnum, args


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, padding_mode='zeros', act=nn.ReLU,
                 last=True, drop_path=0.0):
        super().__init__()
        bias = False
        if norm_layer is None:
            bias = True
            norm_layer = nn.Identity
        self.conv1 = Conv3x3(inplanes, planes, stride, padding_mode=padding_mode, bias=bias)
        self.bn1 = norm_layer(planes)
        self.relu1 = act()
        self.conv2 = Conv3x3(planes, planes, padding_mode=padding_mode, bias=bias)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if last:
            self.relu2 = act()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.drop_path(out) + identity
        if self.last:
            out = self.relu2(out)
        return out


class UBNet(nn.Module):
    def __init__(self, inplanes, dplanes=1, norm_layer=nn.BatchNorm2d, padding_mode='zeros', act=nn.ReLU,
                 blocknum=2, uplayer=UpCC, depth=1, block=BasicBlock, drop_path=0.):
        super().__init__()
        self._norm_layer = norm_layer
        self._act = act
        self._padding_mode = padding_mode
        bc = inplanes // 2
        self.inplanes = bc * 2
        self._img_dpc = 0
        self._img_dprs = np.linspace(0, drop_path, blocknum * (1 + depth))
        encoder_list = [nn.Sequential(
            Basic2d(inplanes + dplanes, bc * 2, norm_layer=norm_layer, kernel_size=3, padding=1,
                    padding_mode=padding_mode, act=act),
            self._make_layer(block, bc * 2, blocknum, stride=1)
        ), ]
        decoder_list = []
        in_channels = bc * 2
        for i in range(depth):
            self.inplanes = in_channels
            out_channels = min(in_channels * 2, 256)
            encoder_list.append(self._make_layer(block, out_channels, blocknum, stride=2))
            decoder_list.append(uplayer(out_channels, in_channels, in_channels, norm_layer=norm_layer))
            in_channels = min(in_channels * 2, 256)
        self.encoder = nn.ModuleList(encoder_list)
        self.decoder = nn.ModuleList(decoder_list)

    def forward(self, x, d):
        feat = []
        if d is not None:
            x = torch.cat([x, d], dim=1)
        for layer in self.encoder:
            x = layer(x)
            feat.append(x)
        out = feat[-1]
        for idx in range(len(feat) - 2, -1, -1):
            out = self.decoder[idx](out, feat[idx])
        return out

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        act = self._act
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
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer, act=act, padding_mode=padding_mode,
                            drop_path=self._img_dprs[self._img_dpc]))
        self._img_dpc += 1
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, act=act, padding_mode=padding_mode,
                                drop_path=self._img_dprs[self._img_dpc]))
            self._img_dpc += 1

        return nn.Sequential(*layers)


class Permute(nn.Module):
    def __init__(self, in_channels, out_channels=1, stride=2, norm_layer=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.stride = stride
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            Basic2d(in_channels=in_channels, out_channels=in_channels, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=in_channels, out_channels=in_channels, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Conv1x1(in_channels, out_channels * stride ** 2, bias=True),
            Rearrange('b (c h2 w2) h w -> b c (h h2) (w w2)', c=out_channels, h2=stride, w2=stride),
        )

    def forward(self, x):
        """
        """
        fout = self.conv(x)
        return fout


class WPool(nn.Module):
    def __init__(self, in_ch, level, drift=1e6):
        super().__init__()
        self.level = level
        self.drift = drift
        self.permute = Permute(in_ch, stride=2 ** level)

    def forward(self, S, fout):
        W = self.permute(fout)
        size = int(2 ** self.level)
        M = (S > 1e-3).float()
        with torch.no_grad():
            maxW = F.max_pool2d((W + self.drift) * M, size, stride=[size, size]) - self.drift
            maxW = F.upsample_nearest(maxW, scale_factor=size) * M
        expW = torch.exp(W * M - maxW) * M
        avgS = F.avg_pool2d(S * expW, kernel_size=size, stride=size)
        avgexpW = F.avg_pool2d(expW, kernel_size=size, stride=size)
        Sp = avgS / (avgexpW + 1e-6)
        return Sp


class UpCat(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, kernel_size=3, padding=1,
                 padding_mode='zeros', act=nn.ReLU):
        super().__init__()
        self.upf = Basic2dTrans(in_channels + 1, out_channels, norm_layer=norm_layer, act=act)
        self.conv = Basic2d(out_channels * 2, out_channels,
                            norm_layer=norm_layer, kernel_size=kernel_size,
                            padding=padding, padding_mode=padding_mode, act=act)

    def forward(self, y, x, d):
        """
        x is
        """
        fout = self.upf(torch.cat([x, d], dim=1))
        fout = self.conv(torch.cat([fout, y], dim=1))
        return fout


class Ident(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *args):
        return args[0]


class Prop(nn.Module):
    """
    """

    def __init__(self, Cfi, Cfp=3, Cfo=2, act=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        """
        """
        self.dist = lambda x: (x * x).sum(1)
        Ct = Cfo + Cfi + Cfi + Cfp
        self.convXF = nn.Sequential(
            Basic2d(in_channels=Ct, out_channels=Cfi, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=Cfi, out_channels=Cfi, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
        )
        self.convXL = nn.Sequential(
            Basic2d(in_channels=Cfi, out_channels=Cfi, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=Cfi, out_channels=Cfi, norm_layer=norm_layer, act=nn.Identity, kernel_size=1,
                    padding=0),
        )
        self.act = act()
        self.coef = Coef(Cfi, 3)

    def forward(self, If, Pf, Ofnum, args):
        """
        """
        num = args.shape[-2]
        B, Cfi, H, W = If.shape
        N = H * W
        B, Cfp, M = Pf.shape
        If = If.view(B, Cfi, 1, N)
        Pf = Pf.view(B, Cfp, 1, M)
        Ifnum = If.expand(B, Cfi, num, N)  ## Ifnum is BxCfixnumxN
        IPfnum = torch.gather(
            input=If.expand(B, Cfi, num, N),
            dim=-1,
            index=args.view(B, 1, num, N).expand(B, Cfi, num, N))  ## IPfnum is BxCfixnumxN
        Pfnum = torch.gather(
            input=Pf.expand(B, Cfp, num, M),
            dim=-1,
            index=args.view(B, 1, num, N).expand(B, Cfp, num, N))  ## Pfnum is BxCfpxnumxN
        X = torch.cat([Ifnum, IPfnum, Pfnum, Ofnum], dim=1)
        XF = self.convXF(X)
        XF = self.act(XF + self.convXL(XF))
        Alpha, Beta, Omega = self.coef(XF)
        Omega = torch.softmax(Omega, dim=2)
        dout = torch.sum(((Alpha + 1) * Pfnum[:, -1:] + Beta) * Omega, dim=2, keepdim=True)
        return dout.view(B, 1, H, W)


class PMP(nn.Module):
    """
    Pre+MF+Post
    """

    def __init__(self, level, in_ch, out_ch, drop_path, up=True, pool=True):
        super().__init__()
        self.level = level
        if up:
            self.upcat = UpCat(in_ch, out_ch)
        else:
            self.upcat = Ident()
        if pool:
            self.wpool = WPool(out_ch, level=level)
        else:
            self.wpool = Ident()
        self.dist = Dist(num=4)
        self.prop = Prop(out_ch)
        self.fuse = UBNet(out_ch, dplanes=3, blocknum=2, depth=5 - level, drop_path=drop_path)
        self.conv = Conv3x3(out_ch, 1, bias=True)
        self.cspn = CSPN(out_ch, pt=2 * (6 - level))

    def pinv(self, S, K, xx, yy):
        fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
        S = S.view(S.shape[0], 1, -1)
        xx = xx.reshape(1, 1, -1)
        yy = yy.reshape(1, 1, -1)
        Px = S * (xx - cx) / fx
        Py = S * (yy - cy) / fy
        Pz = S
        Pxyz = torch.cat([Px, Py, Pz], dim=1).contiguous()
        return Pxyz

    def forward(self, fout, dout, XI, S, K):
        fout = self.upcat(XI, fout, dout)
        Sp = self.wpool(S, fout)
        Kp = K.clone()
        Kp[:, :2] = Kp[:, :2] / 2 ** self.level
        B, _, height, width = Sp.shape
        xx, yy = torch.meshgrid(torch.arange(width, device=Sp.device), torch.arange(height, device=Sp.device),
                                indexing='xy')
        ###############################################################
        # Pre
        Pxyz = self.pinv(Sp, Kp, xx, yy)
        Ofnum, args = self.dist(Sp, xx, yy)
        dout = self.prop(fout, Pxyz, Ofnum, args)
        ###############################################################
        # MF
        Pxyz = self.pinv(dout, Kp, xx, yy).view(dout.shape[0], 3, dout.shape[2], dout.shape[3])
        fout = self.fuse(fout, Pxyz)
        res = self.conv(fout)
        dout = dout + res
        ###############################################################
        # Post
        dout = self.cspn(fout, dout, Sp)
        return fout, dout

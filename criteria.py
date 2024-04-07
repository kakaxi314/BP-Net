#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    criteria.py
# @Project:     BP-Net
# @Author:      jie
# @Time:        2021/3/14 7:51 PM

import torch
import torch.nn as nn

__all__ = [
    'RMSE',
    'MSMSE',
    'MetricALL',
]


class RMSE(nn.Module):

    def __init__(self, mul_factor=1.):
        super().__init__()
        self.mul_factor = mul_factor
        self.metric_name = [
            'RMSE',
        ]

    def forward(self, outputs, target):
        outputs = outputs / self.mul_factor
        target = target / self.mul_factor
        val_pixels = (target > 1e-3).float()
        err = (target * val_pixels - outputs * val_pixels) ** 2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.sqrt(loss / cnt).mean(),


class MSMSE(nn.Module):
    """
    Multi-Scale MSE
    """

    def __init__(self, deltas=(2 ** (-5 * 2), 2 ** (-4 * 2), 2 ** (-3 * 2), 2 ** (-2 * 2), 2 ** (-1 * 2), 1)):
        super().__init__()
        self.deltas = deltas

    def mse(self, est, gt):
        valid = (gt > 1e-3).float()
        loss = est * valid - gt * valid
        return (loss ** 2).mean()

    def forward(self, outputs, target):
        loss = [delta * self.mse(ests, target) for ests, delta in zip(outputs, self.deltas)]
        return loss


class MetricALL(nn.Module):
    def __init__(self, mul_factor):
        super().__init__()
        self.t_valid = 0.0001
        self.mul_factor = mul_factor
        self.metric_name = [
            'RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3', 'D102', 'D105', 'D110'
        ]

    def forward(self, pred, gt):
        with torch.no_grad():
            pred = pred.detach() / self.mul_factor
            gt = gt.detach() / self.mul_factor
            pred_inv = 1.0 / (pred + 1e-8)
            gt_inv = 1.0 / (gt + 1e-8)

            # For numerical stability
            mask = gt > self.t_valid
            # num_valid = mask.sum()
            B = mask.size(0)
            num_valid = torch.sum(mask.view(B, -1), -1, keepdim=True)

            # pred = pred[mask]
            # gt = gt[mask]
            pred = pred * mask
            gt = gt * mask

            # pred_inv = pred_inv[mask]
            # gt_inv = gt_inv[mask]
            pred_inv = pred_inv * mask
            gt_inv = gt_inv * mask

            # pred_inv[pred <= self.t_valid] = 0.0
            # gt_inv[gt <= self.t_valid] = 0.0

            # RMSE / MAE
            diff = pred - gt
            diff_abs = torch.abs(diff)
            diff_sqr = torch.pow(diff, 2)

            rmse = torch.sum(diff_sqr.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            rmse = torch.sqrt(rmse)

            mae = torch.sum(diff_abs.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

            # iRMSE / iMAE
            diff_inv = pred_inv - gt_inv
            diff_inv_abs = torch.abs(diff_inv)
            diff_inv_sqr = torch.pow(diff_inv, 2)

            irmse = torch.sum(diff_inv_sqr.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            irmse = torch.sqrt(irmse)

            imae = torch.sum(diff_inv_abs.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

            # Rel
            rel = diff_abs / (gt + 1e-8)
            rel = torch.sum(rel.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

            # delta
            r1 = gt / (pred + 1e-8)
            r2 = pred / (gt + 1e-8)
            ratio = torch.max(r1, r2)

            ratio = torch.max(ratio, 10000 * (1 - mask.float()))

            del_1 = (ratio < 1.25).type_as(ratio)
            del_2 = (ratio < 1.25 ** 2).type_as(ratio)
            del_3 = (ratio < 1.25 ** 3).type_as(ratio)
            del_102 = (ratio < 1.02).type_as(ratio)
            del_105 = (ratio < 1.05).type_as(ratio)
            del_110 = (ratio < 1.10).type_as(ratio)

            del_1 = torch.sum(del_1.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_2 = torch.sum(del_2.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_3 = torch.sum(del_3.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_102 = torch.sum(del_102.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_105 = torch.sum(del_105.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_110 = torch.sum(del_110.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

            result = [rmse, mae, irmse, imae, rel, del_1, del_2, del_3, del_102, del_105, del_110]

        return result

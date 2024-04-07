#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    augs.py
# @Project:     BP-Net
# @Author:      jie
# @Time:        2021/3/14 8:27 PM


import numpy as np

__all__ = [
    'Compose',
    'Norm',
    'Jitter',
    'Flip',
]


class Compose(object):
    """
    Sequential operations on input images, (i.e. rgb, lidar and depth).
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb, lidar, depth, K_cam):
        for t in self.transforms:
            rgb, lidar, depth, K_cam = t(rgb, lidar, depth, K_cam)
        return rgb, lidar, depth, K_cam


class Norm(object):
    """
    normalize rgb image.
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, rgb, lidar, depth, K_cam):
        rgb = (rgb - self.mean) / self.std
        return rgb, lidar, depth, K_cam

class Jitter(object):
    """
    borrow from https://github.com/kujason/avod/blob/master/avod/datasets/kitti/kitti_aug.py
    """

    def __call__(self, rgb, lidar, depth, K_cam):
        pca = compute_pca(rgb)
        rgb = add_pca_jitter(rgb, pca)
        return rgb, lidar, depth, K_cam



class Flip(object):
    """
    random horizontal flip of images.
    """

    def __call__(self, rgb, lidar, depth, CamK):
        width = rgb.shape[1]
        flip = bool(np.random.randint(2))
        if flip:
            rgb = rgb[:, ::-1, :]
            lidar = lidar[:, ::-1, :]
            depth = depth[:, ::-1, :]
            CamK[0, 2] = (width - 1) - CamK[0, 2]
        return rgb, lidar, depth, CamK


def compute_pca(image):
    """
    calculate PCA of image
    """

    reshaped_data = image.reshape(-1, 3)
    reshaped_data = (reshaped_data / 255.0).astype(np.float32)
    covariance = np.cov(reshaped_data.T)
    e_vals, e_vecs = np.linalg.eigh(covariance)
    pca = np.sqrt(e_vals) * e_vecs
    return pca


def add_pca_jitter(img_data, pca):
    """
    add a multiple of principle components with Gaussian noise
    """
    new_img_data = np.copy(img_data).astype(np.float32) / 255.0
    magnitude = np.random.randn(3) * 0.1
    noise = (pca * magnitude).sum(axis=1)

    new_img_data = new_img_data + noise
    np.clip(new_img_data, 0.0, 1.0, out=new_img_data)
    new_img_data = (new_img_data * 255).astype(np.uint8)

    return new_img_data


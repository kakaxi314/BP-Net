"""
    Berrow from CompletionFormer https://github.com/youmi-zym/CompletionFormer
    ======================================================================

    NYU Depth V2 Dataset Helper
"""

import os
import warnings

import numpy as np
import json
import h5py

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import glob

__all__ = [
    'NYU',
]


class BaseDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    # A workaround for a pytorch bug
    # https://github.com/pytorch/vision/issues/2194
    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)


class NYU(BaseDataset):
    def __init__(self, mode, path='../datas/nyudepthv2', num_sample=500, mul_factor=1., num_mask=8, scale_kcam=False,
                 rand_scale=True, *args, **kwargs):
        super(NYU, self).__init__(None, mode)

        self.mode = mode
        self.num_sample = num_sample
        self.num_mask = num_mask
        self.mul_factor = mul_factor
        self.scale_kcam = scale_kcam
        self.rand_scale = rand_scale

        if mode != 'train' and mode != 'val':
            raise NotImplementedError

        height, width = (240, 320)
        crop_size = (228, 304)

        self.height = height
        self.width = width
        self.crop_size = crop_size

        self.Kcam = torch.from_numpy(np.array(
            [
                [5.1885790117450188e+02, 0, 3.2558244941119034e+02],
                [0, 5.1946961112127485e+02, 2.5373616633400465e+02],
                [0, 0, 1.],
            ], dtype=np.float32
        )
        )

        base_dir = path

        self.sample_list = list(sorted(glob.glob(os.path.join(base_dir, mode, "**/**.h5"))))

    def __len__(self):
        if self.mode == 'train':
            return len(self.sample_list)
        elif self.mode == 'val':
            return self.num_mask * len(self.sample_list)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.mode == 'val':
            seed = idx % self.num_mask
            idx = idx // self.num_mask

        path_file = self.sample_list[idx]

        f = h5py.File(path_file, 'r')
        rgb_h5 = f['rgb'][:].transpose(1, 2, 0)
        dep_h5 = f['depth'][:]

        rgb = Image.fromarray(rgb_h5, mode='RGB')
        dep = Image.fromarray(dep_h5.astype('float32'), mode='F')

        Kcam = self.Kcam.clone()

        if self.mode == 'train':
            if self.rand_scale:
                _scale = np.random.uniform(1.0, 1.5)
            else:
                _scale = 1.0
            scale = int(self.height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)
                Kcam[0, 2] = rgb.width - 1 - Kcam[0, 2]

            rgb = TF.rotate(rgb, angle=degree)
            dep = TF.rotate(dep, angle=degree)

            t_rgb = T.Compose([
                T.Resize(scale),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            t_dep = T.Compose([
                T.Resize(scale),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

            if self.scale_kcam:
                Kcam[:2] = Kcam[:2] * _scale

        else:
            t_rgb = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_dep = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

        if self.mode == 'train':
            dep_sp = self.get_sparse_depth(dep, self.num_sample)
        elif self.mode == 'val':
            dep_sp = self.mask_sparse_depth(dep, self.num_sample, seed)
        else:
            raise NotImplementedError

        rgb = TF.pad(rgb, padding=[8, 14], padding_mode='edge')
        # rgb = TF.pad(rgb, padding=[8, 14], padding_mode='constant')
        dep_sp = TF.pad(dep_sp, padding=[8, 14], padding_mode='constant')
        dep = TF.pad(dep, padding=[8, 14], padding_mode='constant')

        Kcam[:2] = Kcam[:2] / 2.
        Kcam[0, 2] += 8 - 8
        Kcam[1, 2] += -6 + 14

        dep_sp *= self.mul_factor
        dep *= self.mul_factor
        return rgb, dep_sp, Kcam, dep

    def get_sparse_depth(self, dep, num_sample):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel * height * width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)

        if num_idx == 0:
            dep_sp[:, 20:-20:10, 20:-20:10] = 3.

        return dep_sp

    def mask_sparse_depth(self, dep, num_sample, seed):
        channel, height, width = dep.shape
        dep = dep.numpy().reshape(-1)
        np.random.seed(seed)
        index = np.random.choice(height * width, num_sample, replace=False)
        dep_sp = np.zeros_like(dep)
        dep_sp[index] = dep[index]
        dep_sp = dep_sp.reshape(channel, height, width)
        dep_sp = torch.from_numpy(dep_sp)
        return dep_sp

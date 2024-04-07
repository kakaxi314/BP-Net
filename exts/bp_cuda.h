//
// Created by jie on 12/8/22.
//

#ifndef BP_CUDA_H
#define BP_CUDA_H

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <tuple>
#include <iostream>


void dist(
        at::Tensor Pc,
        at::Tensor IPCnum,
        at::Tensor args,
        int H,
        int W
);

at::Tensor Conv2dLocal_F(
        at::Tensor a,
        at::Tensor b
);

std::tuple <at::Tensor, at::Tensor> Conv2dLocal_B(
        at::Tensor a,
        at::Tensor b,
        at::Tensor gc
);

void Dist_Cuda(at::Tensor Pc, at::Tensor IPCnum, at::Tensor args,
                 size_t B, size_t Cc, size_t N, size_t M, size_t num, size_t H, size_t W);

void Conv2d_LF_Cuda(at::Tensor x, at::Tensor y, at::Tensor z, size_t N1, size_t N2, size_t Ci, size_t Co, size_t B,
                    size_t K);

void
Conv2d_LB_Cuda(at::Tensor x, at::Tensor y, at::Tensor gx, at::Tensor gy, at::Tensor gz, size_t N1, size_t N2, size_t Ci,
               size_t Co, size_t B, size_t K);


#endif

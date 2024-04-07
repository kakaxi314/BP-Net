//
// Created by jie on 09/02/19.
//
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <tuple>
#include <iostream>
#include "bp_cuda.h"


void dist(
        at::Tensor Pc,
        at::Tensor IPCnum,
        at::Tensor args,
        int H,
        int W
) {
    int B, Cc, N, M, num;
    B = Pc.size(0);
    Cc = Pc.size(1);
    M = Pc.size(2);
    num = args.size(1);
    N = args.size(2);
    Dist_Cuda(Pc, IPCnum, args, B, Cc, N, M, num, H, W);
}


at::Tensor Conv2dLocal_F(
        at::Tensor a,
        at::Tensor b
) {
    int N1, N2, Ci, Co, K, B;
    B = a.size(0);
    Ci = a.size(1);
    N1 = a.size(2);
    N2 = a.size(3);
    Co = Ci;
    K = sqrt(b.size(1) / Co);
    auto c = at::zeros_like(a);
    Conv2d_LF_Cuda(a, b, c, N1, N2, Ci, Co, B, K);
    return c;
}


std::tuple <at::Tensor, at::Tensor> Conv2dLocal_B(
        at::Tensor a,
        at::Tensor b,
        at::Tensor gc
) {
    int N1, N2, Ci, Co, K, B;
    B = a.size(0);
    Ci = a.size(1);
    N1 = a.size(2);
    N2 = a.size(3);
    Co = Ci;
    K = sqrt(b.size(1) / Co);
    auto ga = at::zeros_like(a);
    auto gb = at::zeros_like(b);
    Conv2d_LB_Cuda(a, b, ga, gb, gc, N1, N2, Ci, Co, B, K);
    return std::make_tuple(ga, gb);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("Dist", &dist, "calculate distance on 2D image");
m.def("Conv2dLocal_F", &Conv2dLocal_F, "Conv2dLocal Forward (CUDA)");
m.def("Conv2dLocal_B", &Conv2dLocal_B, "Conv2dLocal Backward (CUDA)");
}


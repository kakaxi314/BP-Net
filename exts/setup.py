from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='BpOps',
    ext_modules=[
        CUDAExtension('BpOps',
                      [
                          'bp_cuda.cpp',
                          'bp_cuda_kernel.cu',
                      ],
                      extra_compile_args={'cxx': ['-g'],
                                          'nvcc': ['-O3']}
                      ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++11', '-ffast-math']

include_dirs = []
library_dirs = []

nvcc_args = [
    #'-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    #'-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_70,code=compute_70'
]

setup(
    name='mesh2sdf',
    ext_modules=[
        CUDAExtension('mesh2sdf', [
            'mesh2sdf_kernel.cu'
        ],
        include_dirs = include_dirs,
        library_dirs = library_dirs,
        extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
    

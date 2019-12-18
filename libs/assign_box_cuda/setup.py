from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='assign_box_cuda',
    ext_modules=[
        CUDAExtension(
            'assign_box_cuda', 
            ['assign_box_cuda.cpp', 'assign_box_kernel.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
})

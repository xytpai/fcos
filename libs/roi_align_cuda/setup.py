from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='roi_align_cuda',
    ext_modules=[
        CUDAExtension(
            'roi_align_cuda', 
            ['roi_align_cuda.cpp', 'roi_align_kernel.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
})

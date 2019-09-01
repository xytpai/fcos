from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sigmoid_focal_loss_cuda',
    ext_modules=[
        CUDAExtension(
            'sigmoid_focal_loss_cuda', 
            ['sigmoid_focal_loss.cpp', 'sigmoid_focal_loss_cuda.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
})

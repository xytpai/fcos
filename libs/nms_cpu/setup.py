from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='nms_cpu',
    ext_modules=[
        CppExtension(
            'nms_cpu', 
            ['nms_cpu.cpp'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
})
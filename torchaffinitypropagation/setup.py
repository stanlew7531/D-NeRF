from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME
from torch.utils.cpp_extension import CppExtension, CUDAExtension

# In any case, include the CPU version
modules = [
    CppExtension('torchaffinitypropagation.cpu',
                 ['src/cpu/affinitypropagation_cpu.cpp']),
]

# If nvcc is available, add the CUDA extension
# TODO: implement the cuda kernel and remove False
if CUDA_HOME and False:
    modules.append(
        CUDAExtension('affinitypropagation.cuda',
                      ['src/cuda/affinitypropagation_cuda.cpp',
                       'src/cuda/affinitypropagation_cuda_kernel.cu'])
    )

tests_require = [
    'pytest',
]

# Now proceed to setup
setup(
    name='torchaffinitypropagation',
    version='1.0',
    description='An affinity propagation implementation for pytorch',
    keywords='affinity propogation',
    author='Stanley Lewis',
    author_email='stanlew@umich.edu',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    ext_modules=modules,
    tests_require=tests_require,
    extras_require={
        'test': tests_require,
    },
    cmdclass={
        'build_ext': BuildExtension
    }
)

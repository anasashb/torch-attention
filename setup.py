from setuptools import setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

extensions = [
    CppExtension(
        name="adlers.nlp.causal_product_cpu",
        sources=["src/adlers/csrc/causal_product_cpu.cpp"],
        extra_compile_args=["-fopenmp", "-ffast-math"],
        extra_link_args=["-fopenmp"],
    )
]

if CUDA_HOME is not None:
    extensions.append(
        CUDAExtension(
            name="adlers.nlp.causal_product_cuda",
            sources=["src/adlers/csrc/cuda/causal_product_cuda.cu"],
        )
    )

setup(
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension},
)

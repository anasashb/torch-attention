from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    ext_modules=[
        CppExtension(
            name="adlers.nlp.causal_product_cpu",
            sources=["src/adlers/csrc/causal_product_cpu.cpp"],
            extra_compile_args=["-fopenmp", "-ffast-math"],
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

from os.path import abspath, dirname, join
from setuptools import setup, find_packages, Extension

from torch.utils import cpp_extension

project_root = dirname(abspath(__file__))

requirements = [
    "numpy",
    "torch>=1.13.0",
    "safetensors",
    "transformers>=4.26.1",
    "triton>=2.0.0"
]

extras_require = {
    "llama": ["transformers>=4.28.0"]
}

setup(
    name="auto_gptq",
    packages=find_packages(),
    version="v0.0.1-dev",
    install_requires=requirements,
    extras_require=extras_require,
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)

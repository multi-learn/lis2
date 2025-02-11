from setuptools import find_packages
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="DeepFilaments",
    version="1.0.0",
    description="Deep learning for filament extraction",
    author="Loris Berthelot",
    author_email="loris.berthelot@lis-lab.fr",
    url="https://gitlab.lam.fr/bigsf/deepfilaments",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=[
    ],
    packages=find_packages(where=".", exclude=("tests",)),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "h5py",
        "astropy",
        "reproject",
        "scikit-image",
        "scikit-learn",
        "pandas",
        "torch",
        "seaborn",
        "openpyxl",
        "numba",
        "networkx",
        "tqdm",
        "einops",
        "timm",
        "configurable-cl",
    ],
    python_requires=">=3.8",
)

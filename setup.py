# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

import unirec

name = "unirec"
version = unirec.version    # version should be updated in `unirec/__init__.py`
author = "Jianxun Lian"
author_email = "jialia@microsoft.com"
description = "A compact recommender library for universal recommendation systems"

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()


url = "https://github.com/microsoft/UniRec"    # usually the link of the repo
packages = [pack for pack in setuptools.find_packages() if pack.startswith("unirec")]

# more optional classifiers in https://pypi.org/classifiers/
classifiers = [
        "Development Status :: 3 - Alpha",  # optional
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Environment :: GPU :: NVIDIA CUDA",   # optional
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Natural Language :: English",   # optional
        "License :: OSI Approved :: MIT License",   # select a license if upload to PyPi
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]


# dependency
install_requires = [
    "torch>=1.10.0,<=1.13.1",
    "debugpy>=1.6.3", 
    "feather-format>=0.4.1", 
    "numba>=0.56.3", 
    "pandas>=1.5.1", 
    "pyarrow>=10.0.0", 
    "scikit-learn>=1.1.3", 
    "scipy>=1.9.3", 
    "setproctitle>=1.3.2", 
    "swifter>=1.3.4", 
    "tqdm>=4.64.1", 
    "wandb>=0.14.2", 
    "tensorboard>=2.13.0", 
    "cvxpy>=1.3.1", 
    "onnxruntime>=1.15.1", 
    "accelerate>=0.20.3", 
    "pytest>=7.3.1"
]


setuptools.setup(
    name = name,
    version = version,
    author = author,
    author_email = author_email,
    description = description,
    keywords="recommender system, deep learning, pytorch",
    license="MIT",
    long_description = long_description,
    long_description_content_type = "text/markdown",  # if the README file is not txt or markdown file, please change it
    url = url,
    packages = packages,
    classifiers = classifiers,
    python_requires = ">=3.8",
    include_package_data=True,  # necessary for non-python scripts, such as yaml file
    install_requires = install_requires,
)
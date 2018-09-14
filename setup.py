"""mihfepe package definition and install configuration"""

from setuptools import find_packages, setup

setup(
    name="mihifepe",
    version="0.1dev",
    description="Model Interpretability via Hierarchical Feature Perturbation",
    url="https://github.com/cloudbopper/mihifepe",
    author="Akshay Sood",
    license="MIT",
    long_description=open("README.rst").read(),
    packages=find_packages(),
    python_requires=">= 3.5",
    install_requires=[
        "anytree~=2.4.3",
        "h5py~=2.8.0",
        "numpy~=1.15.0",
        "pyhashxx~=0.1.3",
        "scikit-learn~=0.19.2",
        "scipy~=1.1.0"
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]

)

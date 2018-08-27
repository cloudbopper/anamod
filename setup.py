"""mihfepe package definition and install configuration"""

from setuptools import find_packages, setup

setup(
    name="mihifepe",
    version="0.1dev",
    description="Model Interpretability via Hierarchical Feature Perturbation",
    url="https://github.com/cloudbopper/mihifepe",
    author="Akshay Sood",
    license="MIT",
    long_description=open("README.md").read(),
    packages=find_packages(),

    # Requirements
    python_requires=">= 3.5",
    install_requires=[
        "anytree~=2.4.3",
        "h5py~=2.8.0",
        "numpy~=1.15.0",
        "pyhashxx~=0.1.3",
        "rpy2~=2.8.4",
        "scikit-learn~=0.19.2",
        "scipy~=1.1.0"
    ]
)

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
    python_requires=">= 3.6",
    install_requires=[
        "anytree>=2.4.3"
        "numpy>=1.12.1",
        "scikit-learn>=0.18.1"
    ]
)

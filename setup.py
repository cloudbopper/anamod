"""mihifepe package definition and install configuration"""

from setuptools import find_packages, setup

# pylint: disable = invalid-name

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

setup(
    author="Akshay Sood",
    author_email='sood.iitd@gmail.com',
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
    ],
    description="Model Interpretability via Hierarchical Feature Perturbation",
    entry_points={
        "console_scripts": [
            "mihifepe=mihifepe.master:main",
        ],
    },
    include_package_data=True,
    install_requires=[
        "anytree~=2.4.3",
        "h5py~=2.8.0",
        "numpy~=1.15.0",
        "pyhashxx~=0.1.3",
        "scikit-learn~=0.19.2",
        "scipy~=1.1.0"
    ],
    keywords="mihifepe",
    license="MIT",
    long_description=readme + "\n\n" + history,
    name="mihifepe",
    packages=find_packages(include=["mihifepe"]),
    python_requires=">= 3.5",
    url="https://github.com/cloudbopper/mihifepe",
    version="0.1.0dev",
    zip_safe=False,
)

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="nwflex._cython.nwflex_dp",
        sources=["nwflex/_cython/nwflex_dp.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="nwflex",
    version="0.1.0",  
    author="Zhezhen Yu and Dan Levy",
    author_email="zhezhen.yu.github@gmail.com, danlevy.github@gmail.com",
    description="Python package for the NW-flex algorithm",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nwflex/nwflex",
    packages=find_packages(),
    ext_modules=cythonize(extensions, language_level=3),

    install_requires=[
        "numpy"
    ],
    extras_require={
        "plot": [
            "matplotlib",
            "seaborn",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
        "notebooks": [
            "jupyter",            
            "nbconvert",
            "nbmerge",
            "nbformat",
            "pandas",
            "matplotlib",
            "seaborn",
            "pandoc",
        ],
        "all": [
            "matplotlib",
            "seaborn",
            "pytest>=7.0",
            "pytest-cov",
            "jupyter",            
            "nbconvert",
            "nbmerge",
            "nbformat",
            "pandas",
            "pandoc",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.11',
)
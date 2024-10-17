from setuptools import find_packages,setup

setup(
    name="neuronflow",
    version="0.1.0",
    description="A lightweight machine learning library",
    author="Riddhick Dalal",
    author_email="riddhick14@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    python_requires='>=3.6',
)
from setuptools import find_packages,setup

setup(
    name="neuronflow",
    version="0.0.1",
    description="A lightweight machine learning library",
    author="Riddhick Dalal",
    author_email="riddhick14@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.6',
)
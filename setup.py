from setuptools import find_packages,setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="neuronflow",
    version="0.0.4",
    long_description=long_description,
    long_description_content_type='text/markdown',
    #description="A lightweight machine learning library",
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
<h1 align="center">🔮 NeuronFlow 🔮</h1>

<p align="center">
    <img src="https://img.shields.io/badge/python-3.x-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Build Status">
    <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License">
</p>

<p align="center"><i>A lightweight machine learning library written in Python.</i></p>

---

# 🌌 Overview

**NeuronFlow** is a Python package designed for building, training, and evaluating machine learning models. Whether you're a beginner or a seasoned professional, this library provides tools for quick prototyping and production-level model development.

Features include:
- 🧠 **Customizable Models**: Build custom models from scratch using an intuitive API.
- ⚡ **Optimized for Performance**: Built-in optimizations for faster training.
- 📊 **Evaluation Tools**: Built-in metrics and visualizations for evaluating models.
- 💡 **Explainability**: Model insights for transparency and debugging.


# 🌟 Models Available

- **Linear Models**: 
  - Linear Regression
  - Multiple Linear Regression



  

# 🚀 Installation

```bash
pip install neuronflow

```
# 🧑‍🏫 How To Use

```python
import neuronflow as nf

#Regression
from neuronflow import regerssion

#Linear Regression
X=np.array([1,2,3,4])
Y=np.array([5,6,7,8])
model=nf.regression.linear(X,Y)
model.fit() 
#Inference 
value=model.value(np.array([9,10]))



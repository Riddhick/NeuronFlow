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


# 🌟 Tools Available

- **Linear Models**: 
  - Linear Regression
  - Multiple Linear Regression
  - Polynomial Regression
- **Classifier Models**:
  - Logistic Regression

- **Evaluation Metrics**:
  - Mean Square Error(MSE)
  - Mean Absolute Error(MAE)
  - Root Mean Square Error(RMSE)
  - R2 Score



  

# 🚀 Installation

```bash
pip install neuronflow

```
#  How To Use

```python
import neuronflow as nf

#Regression
from neuronflow import regerssion
from neuronflow import classifier

#Linear Regression
X=np.array([1,2,3,4])
Y=np.array([5,6,7,8])
linear_model=regression.linear(X,Y)
linear_model.fit() 
#Inference 
linera_value=model.value(np.array([9,10]))

#Classification
x=np.array([[0.1,1.2,1.5,2.0,1.0,2.5],[1.1,.9,1.5,1.8,2.5,.5]])
y=np.array([0,0,1,1,1,0])

logistic_model=classifier.logistic(x,y,lr=0.1)
logistic_model.fit()

logistic_model.predict(x)
```


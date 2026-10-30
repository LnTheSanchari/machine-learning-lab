# Linear Regression from Scratch

This module provides a Python implementation of Linear Regression, built from first principles using NumPy.

The objective is to understand the mathematical foundations of linear regression, optimization, and regularization through implementation and experimentation rather than relying entirely on pre-built machine learning libraries.

## Features

The implementation supports:

- **Multiple Features:** Works with datasets containing one or multiple independent variables.
- **Loss Functions:**
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
- **Regularization:**
  - L1 Regularization (Lasso)
  - L2 Regularization (Ridge)
  - No Regularization
- **Gradient-Based Learning:** Learns model parameters using gradient descent.
- **Configurable Hyperparameters:** Allows customization of learning rate, epochs, and regularization strength.
## Mathematical Foundation

The linear regression model is defined as:

$$
\hat{Y} = XW + b
$$

where:

- `X` represents the input feature matrix.
- `W` represents the model weights.
- `b` represents the bias.
- $\hat{Y}$ represents the predicted values.

### Loss Functions

**Mean Squared Error (MSE)**

$$
J(W,b) = \frac{1}{2m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
$$

**Mean Absolute Error (MAE)**

$$
J(W,b) = \frac{1}{m}\sum_{i=1}^{m}|y_i-\hat{y}_i|
$$

### Regularization

**L1 Regularization**

$$
J_{\text{L1}} = J(W,b) + \frac{\lambda}{m} \sum_{j=1}^{n}|w_j|
$$

**L2 Regularization**

$$
J_{\text{L2}} = J(W,b) + \frac{\lambda}{2m} \sum_{j=1}^{n}w_j^2
$$

Here, $\lambda$ controls the regularization strength.
### Regularization

The implementation also explores the effects of regularization on model parameters.

- **L1:** Adds a penalty proportional to the absolute values of the weights.
- **L2:** Adds a penalty proportional to the squared values of the weights.

Regularization helps control model complexity and can reduce overfitting.

## Usage

Import the implementation into your Python notebook:

```python
from LinearModels import LinearRegression
```

Initialize a model by specifying the initial weights and bias:

```python
model = LinearRegression(
    w,
    b,
    loss_func='mse',
    reg=None,
    lambda_=0.01
)
```

Train the model:

```python
model.learn(
    X,
    Y,
    alpha=0.001,
    epochs=10000
)
```

Here:
- `loss_func`: Selects either `'mse'` or `'mae'`.
- `reg`: Selects `'l1'`, `'l2'`, or `None`.
- `lambda_`: Controls regularization strength.
- `alpha`: Learning rate.
- `epochs`: Number of training iterations.

## Experiments

The implementation is accompanied by Jupyter Notebook experiments exploring:

1. Linear Regression using MSE without regularization.
2. Linear Regression using MAE without regularization.
3. Linear Regression using MSE with L1 regularization.
4. Linear Regression using MAE with L2 regularization.

These experiments help visualize how different loss functions and regularization techniques influence the learned regression model.

## Technologies

- Python
- NumPy
- Matplotlib
- Jupyter Notebook

---

**Part of my journey into Machine Learning and Deep Learning from first principles.**

Learning mathematics, implementing algorithms, experimenting with models, and developing intuition one step at a time.

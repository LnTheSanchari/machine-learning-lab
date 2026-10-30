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

Regularization introduces a penalty term into the objective function to control model complexity.

**L1 Regularization**

$$
J_{\text{L1}} = J(W,b) + \frac{\lambda}{m} \sum_{j=1}^{n}|w_j|
$$

**L2 Regularization**

$$
J_{\text{L2}} = J(W,b) + \frac{\lambda}{2m} \sum_{j=1}^{n}w_j^2
$$

Here, $\lambda$ controls the regularization strength.

## Implementation

The implementation is available in `LinearRegression.py`.

### Importing the Model

```python
from LinearModels import LinearRegression
```

### Initializing the Model

```python
model = LinearRegression(
    w,
    b,
    loss_func='mse',
    reg=None,
    lambda_=0.01
)
```

### Training the Model

```python
model.learn(
    X,
    Y,
    alpha=0.001,
    epochs=10000
)
```

### Parameters

| Parameter | Description |
|---|---|
| `w` | Initial weight parameters |
| `b` | Initial bias |
| `loss_func` | Loss function: `'mse'` or `'mae'` |
| `reg` | Regularization: `'l1'`, `'l2'`, or `None` |
| `lambda_` | Regularization strength |
| `alpha` | Learning rate |
| `epochs` | Number of training iterations |

## Experimental Results

The implementation is accompanied by Jupyter Notebook experiments exploring different combinations of loss functions and regularization techniques.

### Comparison of Linear Regression Models

The following experiment compares four configurations:

1. MSE without regularization.
2. MAE without regularization.
3. MSE with L1 regularization.
4. MAE with L2 regularization.

   
<img width="12000" height="7200" alt="comparision_linear_models" src="https://github.com/user-attachments/assets/097653b4-26da-4a18-8ef3-6e7aef819d6b" />


These experiments help visualize how different loss functions and regularization techniques influence the learned regression model.



**Learning by understanding, implementing, and experimenting — one model at a time.**

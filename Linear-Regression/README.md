# Linear Regression from Scratch📊

This module provides a Python implementation of Linear Regression, built from first principles using NumPy. 

To understand the whole theory behind the **Linear Regression Model** and 
1. Simple Reg implementation Using Vectorized function of numpy.
2. Matrix calculus and Linear Algebra view of **MSE**
3. Multiple Reg implementation, for both loss functions **MSE and MAE**.
4. Need of Feature Scaling, with nice contour plots.(**Important**)
5. How can we express the gradient of **MAE** in vectorized form.

To understand the above, check out this link: [Linear Regression with Feature Scaling in Detail](https://github.com/LnTheSanchari/Linear-Regression-From-Scratch/blob/main/Linear_Regression_From_Scratch.ipynb)

This repo mostly focuses on creating a **LinearRegression** Class and compare the results with **sklearn** linear regression.

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

## Feature Scaling⚡

The implementation is accompanied by Jupyter Notebook experiments exploring **Geometric intuition behind the need of Feature scaling**

To put it simply, the primary reason for feature scaling: **It makes loss surface's curvature to be identical in all directions**, which helps the model to learn as quicky as possible( **Learning Rate**).


<img width="12000" height="6000" alt="Max_normalization" src="https://github.com/user-attachments/assets/59eac9bc-b8ed-4e21-b624-0d35df90c7cc" />


## Does More Training Helps the model?
 **It goes with two questions:**
 1. **Is it good to train the model for more and more number of epochs? Does it help the model to uncover the patterns that are generalizable to unseen data?**
 2. **Does more training makes both validation loss and training loss decrease?**

A nice graph in the notebook answer these two questions for us!!.


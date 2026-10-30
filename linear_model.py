# %% Import Datasets and libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# %% Creation of temporary Data.
# rng = np.random.default_rng(seed=17)
# X = rng.uniform(0, 1, size=(20, 2))
# Y = (10*X[::, 0] + 2*X[::, 1]).reshape(-1, 1) + rng.normal(0, 0.1, size=(20, 1)) # Adding some noise to the data.

# %% Check the data shapes.
# print(X.shape, Y.shape, sep='\n')



# %% 

class LinearRegression:
    def __init__(self, w, b, *, loss_func='mse', reg=None, lambda_=0.01):

        # -------------loss and regularization checks----------------------
        assert loss_func in ['mse', 'mae'], "Loss function should be either 'mse' or 'mae'."
        assert reg in [None, 'l1', 'l2'], "Regularization should be either None, 'l1', or 'l2'."
        # -----------weights and bias checks--------------------------------
        assert isinstance(w, np.ndarray), 'Weights vectors must be ndarray'
        assert w.ndim == 2 and w.shape[1] == 1, "w should be a column vector of shape (d, 1)"
        assert isinstance(b, (int, float)), "b should be a scalar value."
        # -------------Regularization parameter checks-----------------------
        assert lambda_ >= 0 , "Regularization Strenght must be non-negative"


        self.loss_func = loss_func
        self.reg = reg
        self.lambda_ = lambda_
        self.w = w
        self.b = b

    def predict(self, X):
        """
        Generate predictions using the current model parameters.

        Parameters
        ----------
        X : numpy.ndarray, shape (d,), (1, d), (d, 1), or (m, d)
            Input features. Supports:
            - 1D array representing a single sample.
            - Row vector representing a single sample.
            - Column vector representing a single sample or multiple
            samples with a single feature, depending on model dimension.
            - 2D array representing multiple samples.

        Returns
        -------
        numpy.ndarray or float
            Predicted values.
            Returns a scalar for a single-sample input and a column
            vector of shape (m, 1) for multiple samples.

        Notes
        -----
        The interpretation of a column vector depends on the number
        of features in the model.
        
        """
        assert X.ndim in [1, 2], "X should be either a row/column vector or a 2D array."
        if X.ndim == 1:
            assert X.shape[0] == self.w.shape[0], "The number of features in X should match the number of weights in w."
            return (X.reshape(1, -1) @ self.w + self.b).item()  # Return a scalar value.
        m, n = X.shape

        if n ==1 :
            if self.w.shape[0]==1:
                # X is arrays of m samples of single feature.
                return X@self.w + self.b
            else:
                # Multiple features.
                assert m==self.w.shape[0], 'The row dimensions must match with number of weight in w.'
                return (X.T @ self.w + self.b).item()
        if m==1: # row vector
            assert n==self.w.shape[0], 'The number of features in X should match the number of weights in w'
            return (X@self.w + self.b).item()

        # Then it leaves us with 2D array, n must match with number of features.
        assert n == self.w.shape[0], 'The number of feature in X should match the number of weights in w'

        return (X@self.w  + self.b)

    def compute_loss(self, X, Y):
        """
        Parameters:
        ----------
        X : n x d data matrix of features.
        Y: n x 1 ground truth values of each data point.

        Return:
        --------
            Compute the specified loss for the given weight assignement.
        
        """

        self.dimension_check(X, Y)
        m, n = X.shape
        loss = 0

        if self.loss_func == 'mse':
            diff = (self.predict(X) - Y)
            loss_mse = (1/(2*m))*((diff**2).sum())
            loss = loss_mse
        else:
            diff = (self.predict(X) - Y)
            loss_mae = (1/m)*(np.abs(diff).sum())
            loss = loss_mae

        if self.reg=='l1':
            loss_reg = (1/m)*self.lambda_*((np.abs(self.w)).sum())
            loss += loss_reg
        elif self.reg=='l2':
            loss_reg = (1/(2*m))*self.lambda_*((self.w**2).sum())
            loss += loss_reg
        return loss
    

    def compute_gradient(self, X, Y):
        self.dimension_check(X, Y)
        m, n = X.shape

        if self.loss_func == 'mse':
            diff = (self.predict(X) - Y)
            dw = (1/m)*(X.T @ diff)
            db = (1/m)*(diff.sum())
        else:
            diff = (self.predict(X) - Y)
            dw = (1/m)*(X.T@(np.sign(diff)))
            db = (1/m)*((np.sign(diff)).sum())

        if self.reg=='l1':
            dw += (1/m)*self.lambda_*(np.sign(self.w))
        elif self.reg=='l2':
            dw += (1/m)*self.lambda_*self.w
        return dw, db

    def learn(self, X, Y, alpha=0.001, epochs=500):

        # --------------Error Checks----------------------------
        self.dimension_check(X, Y)
        assert np.isfinite(alpha) and alpha > 0, 'Learning rate of the model must be positive'
        assert isinstance(epochs, (int, np.integer)) and epochs > 0, 'Epochs must be positive integer'
        # -------------Learning------------------------------------

        loss = lambda: self.compute_loss(X, Y)
        per_step = max(1, int(epochs*0.5))
        loss_hist = []

        for epoch_num in range(1, epochs+1):
            dw, db = self.compute_gradient(X, Y)
            self.w = self.w - alpha*dw
            self.b = self.b - alpha*db

            if epoch_num % per_step == 0:
                print(f"Epoch: {epoch_num}, Loss: {loss()}")
            loss_hist.append(loss())
        return loss_hist
        

    def dimension_check(self, X, Y):
        assert X.ndim == 2, "X should be a 2D array."
        assert Y.ndim == 2 and Y.shape[1] == 1, "Y should be a column vector of shape (m, 1)"
        m, n = X.shape
        assert n == self.w.shape[0], "The number of features in X should match the number of weights in w."
        assert Y.shape[0] == m, "The number of samples in X and Y should match."

    def __repr__(self):
        return (f"LinearRegression(loss_func='{self.loss_func}', "
                f"reg='{self.reg}', lambda_={self.lambda_}, "
                f"n_features={self.w.shape[0]})")


# %%
w = np.ones((2, 1)) # Initial weight assignment.
b = 0.3




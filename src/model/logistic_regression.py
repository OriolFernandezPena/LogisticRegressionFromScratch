import numpy as np
from src.exceptions import NotFittedError

class LogisticRegression():
    def _init__(self):
        '''
            This is a simple implementation of the Logistic Regression
            algorithm from scratch. 
        '''
        self.b = None
        self.theta = None
        self.losses = list()

    def _log_function(self, x):
        '''
            Method to compute the logistic or sigma function
        '''
        return 1 / (1 + np.exp(-x))
    
    def predict_proba(self, X : np.array):
        '''
            Method to predict the probabilities of the arrays in
            X of being y = 1.
        '''
        if self.theta is None:
            raise NotFittedError("This class hasn't been fitted yet")
        
        return self._log_function(np.dot(X, self.theta) + self.b)

    def _loss(self, y_true, y_pred):
        '''
            Method to compute the loss function.
            Input:
                - y_true `np.array`: array containing the actual classes
                - y_pred `np.array`: this should be the output of self.predict_proba
            Output:
                - `float`: loss of that epoch 
        '''
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def _d_loss(self, X, y_true, y_pred):
        '''
            Method to compute the partial derivatives of the loss
            with respect to the coefficients of the logistic regression.
            Input:
                - X `np.array`: observations
                - y_true `np.array`: array containing the actual classes
                - y_pred `np.array`: this should be the output of self.predict_proba
            Output:
                - `np.array`: derivatives of the self.theta coefficients.
                - `float`: derivative of the self.b coefficient.
        '''
        _dif_ys = y_pred - y_true
        m = _dif_ys.shape[0]
        return np.matmul(X.T, _dif_ys) / m, _dif_ys / m
        
    
    def fit(self, X: np.array, y: np.array, epochs: int = 100, eta: float = 0.01, save_losses: bool = False):
        '''
            Method to fit the logistic regression.
            Inputs:
                - X `np.array`: observations.
                - y `np.array`: classes.
                - epochs `int` (optional): number of epochs to run the gradient descent.
                - eta `float` (optional): learning rate.
                - save_losses `bool` (optional): if True it saves the losses of the fit process in 
                    self.losses
        '''
        # Comprobar tamaños compatibles
        # Comprobar datos son numéricos
        # Comprobar que y solo tiene dos etiquetas 0 y 1
        if not isinstance(epochs, int):
            raise ValueError('`epochs` must of type `int`')
        
        self.losses = list()
        _, n_vars = X.shape
        self.theta = np.zeros(n_vars)
        self.b = 0
        
        for _ in range(epochs):
            _predictions = self.predict_proba(X)
            if save_losses:
                self.losses.append(self._loss(y, _predictions))
            d_thetas, d_b = self._d_loss(X, y, _predictions)
            self.theta -= eta * d_thetas
            self.b -= eta * d_b

    def predict(self, X, threshold : float = 0.5):
        '''
            Method to predict the classes:
            Inputs:
                - X `np.array`: observations to predict.
                - threshold `float`: threshold to the probabilities.
        '''
        return (self.predict_proba(X) >= threshold).astype(int)



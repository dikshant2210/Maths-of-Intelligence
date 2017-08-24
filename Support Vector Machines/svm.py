import numpy as np

class SupportVectorClassifier:
    """docstring forSVM."""
    def __init__(self, X, Y, learning_rate=0.001, bias=True):
        self.lr = learning_rate
        self.X = X
        self.Y = Y
        if bias:
            b = np.ones(self.X.shape[0]) * -1
            self.X = self.X.concatenate((self.X, b), axis=1)
        self.W = np.random.randn(len(self.X[0]))

    def computeOutput(self):
        y = np.dot(self.W.T, self.X.T)
        return y

    def computeError(self):
        y = computeOutput()
        error = 1 - y * self.Y
        error_bool = error <= 0
        total_error = np.sum(error * error_bool) / np.float(len(error))
        return total_error

    def gradient(self, reg):
        reg_gradient = 2 * reg * self.W
        cost_gradient = -1 * self.Y * self.X
        cost_gradient_bool = cost_gradient < 1
        total_cost_gradient = np.sum(cost_gradient_bool * cost_gradient, axis=0) / np.float(len(cost_gradient_bool))
        return reg_gradient, total_cost_gradient

    def gradientUpdate(self, iterations=100):
        pass

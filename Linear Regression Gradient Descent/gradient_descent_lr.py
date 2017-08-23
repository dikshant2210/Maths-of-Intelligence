from numpy import *

class LinearRegression(object):
    """docstring for LinearRegression."""
    def __init__(self, x_data, y_data, num_iterations, learning_rate):
        self.x_data = x_data
        self.y_data = y_data
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

    def computeError(self, m, b):
        totalError = 0
        for i in range(0, len(self.x_data)):
            x = self.x_data[i]
            y = self.y_data[i]
            totalError += (y - (m * x + b)) ** 2
        return totalError / float(len(self.x_data))

    def gradient(self, m ,b):
        b_gradient = 0
        m_gradient = 0
        for i in range(0, len(self.x_data)):
            x = self.x_data[i]
            y = self.y_data[i]
            b_gradient += 2 * (y - (m * x + b))
            m_gradient += 2 * -(x) * (y - (m * x + b))
        m_gradient /= len(self.x_data)
        b_gradient /= len(self.x_data)
        return (m_gradient, b_gradient)

    def gradientUpdate(self):
        m = 0
        b = 0
        for i in range(0, self.num_iterations):
            (m_gradient, b_gradient) = self.gradient(m, b)
            m -= (self.learning_rate * m_gradient)
            b -= (self.learning_rate * b_gradient)
        return (m, b)

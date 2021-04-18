import numpy as np


class Helper:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x * 0.03))
    @staticmethod
    def sigmoid_der(x):
        return x * (1 - x)
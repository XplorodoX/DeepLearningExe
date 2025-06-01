import numpy as np

class Base:
    def __init__(self):
        self.trainable = False
        # This member variable will be populated by the optimizer.
        # It is not set by the layer itself.
        self._optimizer = None 

    def forward(self, input_tensor):
        raise NotImplementedError

    def backward(self, error_tensor):
        raise NotImplementedError

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
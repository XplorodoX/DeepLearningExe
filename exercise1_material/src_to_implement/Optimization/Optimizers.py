import numpy as np

class sgd:
    def __init__(self, learning_rate=0.01):
        """
        Initialize the SGD optimizer.
        :param learning_rate: Learning rate for the optimizer.
        """
        self.learning_rate = learning_rate

    def step(self, layer):
        """
        Perform a single optimization step.
        :param layer: The layer to optimize.
        """
        if not layer.trainable:
            return

        # Update weights
        layer.weights -= self.learning_rate * layer.grad_weights
import numpy as np

class Sgd:
    def __init__(self, learning_rate=0.01):
        """
        Initialize the SGD optimizer.
        :param learning_rate: Learning rate for the optimizer.
        """
        # Ensure the learning_rate passed during initialization is assigned
        # to the instance variable self.learning_rate.
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Calculate the update for the weights.
        :param weight_tensor: Current weights of the layer.
        :param gradient_tensor: Gradient of the loss with respect to the weights.
        :return: Updated weights.
        """
        # This line performs the standard SGD update.
        # new_weights = current_weights - learning_rate * gradient
        return weight_tensor - self.learning_rate * gradient_tensor
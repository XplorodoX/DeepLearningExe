import numpy as np

class NeuralNetwork:
    def __init__(self, layers, loss_function, optimizer):
        """
        Initialize the Neural Network.
        :param layers: List of layers in the network.
        :param loss_function: Loss function to be used.
        :param optimizer: Optimizer to be used for weight updates.
        """
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, input_tensor):
        """
        Perform a forward pass through the network.
        :param input_tensor: Input tensor to the network.
        :return: Output tensor from the network.
        """
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
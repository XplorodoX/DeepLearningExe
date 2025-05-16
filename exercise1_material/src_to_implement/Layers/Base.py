import numpy as np

class Base:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True

    def forward(self, input_tensor):
        """
        Forward pass of the layer.
        :param input_tensor: Input tensor to the layer.
        :return: Output tensor from the layer.
        """
        raise NotImplementedError("Forward pass not implemented.")

    def backward(self, grad_output):
        """
        Backward pass of the layer.
        :param grad_output: Gradient of the loss with respect to the output of this layer.
        :return: Gradient of the loss with respect to the input of this layer.
        """
        raise NotImplementedError("Backward pass not implemented.")

    def update(self, learning_rate):
        """
        Update the weights and biases of the layer.
        :param learning_rate: Learning rate for the update.
        """
        raise NotImplementedError("Update method not implemented.")
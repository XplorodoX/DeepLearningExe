import numpy as np

class SoftMax:
    def __init__(self):
        """
        Initialize the SoftMax layer.
        """
        self.trainable = False

    def forward(self, input_tensor):
        """
        Compute the softmax of the input tensor x.
        :param x: Input tensor
        :return: Softmax output
        """
        # Subtract the max for numerical stability
        e_x = np.exp(input_tensor - np.max(input_tensor))
        return e_x / e_x.sum(axis=0, keepdims=True)

    def backward(self, error_tensor):
        """
        Compute the gradient of the softmax function.
        :param grad_output: Gradient of the loss with respect to the output
        :return: Gradient of the loss with respect to the input
        """
        # which returns a tensor that serves as the error tensor for the previous layer.
        return error_tensor
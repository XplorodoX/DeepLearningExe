import numpy as np

class ReLU:
    def __init__(self):
        self.trainable = False
    
    def forward(self, input_tensor):
        """
        Compute the ReLU activation function.
        :param input_tensor: Input tensor       
        :return: ReLU output
        """
        # Apply ReLU activation function
        self.input_tensor = input_tensor
        output = np.maximum(0, input_tensor)
        return output
    
    def backward(self, error_tensor):
        """
        Compute the gradient of the ReLU function.
        :param error_tensor: Gradient of the loss with respect to the output
        :return: Gradient of the loss with respect to the input
        """
        # Compute the gradient of ReLU
        gradient = np.where(self.input_tensor > 0, 1, 0)
        return error_tensor * gradient
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

        # Store the input tensor for backward pass
        self.input_tensor = input_tensor
        # Apply ReLU activation
        output = np.maximum(0, input_tensor) 
        return output
    
    def backward(self, error_tensor):
        """
        Compute the gradient of the ReLU function.
        :param error_tensor: Gradient of the loss with respect to the output
        :return: Gradient of the loss with respect to the input
        """
        # Compute the gradient of ReLU

        gradient = error_tensor.copy()

        return error_tensor*(self.input_tensor > 0)
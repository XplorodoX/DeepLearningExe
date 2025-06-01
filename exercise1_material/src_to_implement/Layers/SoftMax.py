import numpy as np

class SoftMax:
    def __init__(self):
        """
        Initialize the SoftMax layer.
        """
        self.trainable = False
        self.y = None  # Store output for backward pass

    def forward(self, input_tensor):
        """
        Compute the softmax of the input tensor x.
        :param input_tensor: Input tensor
        :return: Softmax output
        """
        # Subtract the max for numerical stability
        stable_input_tensor = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        e = np.exp(stable_input_tensor)
        self.y = e / np.sum(e, axis=1, keepdims=True)  # Softmax computation
        return self.y
    

    def backward(self, error_tensor):
        """
        Compute the gradient of the loss with respect to the input of the softmax layer.

        This implementation follows Equation (14) from the FAU lecture slides:
        E_{n-1} = y_hat * (E_n - sum_{j=1}^{N} (E_{n,j} * y_hat_j)) [cite: 63]

        All operations are element-wise, and the sum is over the classes for each batch item. [cite: 64]
        The computation is performed for every element of the batch. [cite: 63]

        :param error_tensor: E_n, gradient of the loss with respect to the output of this layer (y_hat).
                             Expected shape (N, B) where N is the number of classes and B is the batch size.
        :return: E_{n-1}, gradient of the loss with respect to the input of this layer (x).
                 Shape (N, B).
        """
        sum_error_times_y_hat = np.sum(error_tensor * self.y, axis=1, keepdims=True)

        term_in_parenthesis = error_tensor - sum_error_times_y_hat

        grad_input = self.y * term_in_parenthesis

        return grad_input



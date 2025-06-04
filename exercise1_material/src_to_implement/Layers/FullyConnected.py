import numpy as np
from .Base import Base

class FullyConnected(Base):
    def __init__(self, input_size, output_size):
        """
        Initializes a fully connected layer.

        Args:
            input_size (int): The number of features in the input.
            output_size (int): The number of features in the output.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Better weight initialization
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        
        self.trainable = True
        self._gradient_weights = None
        self._input_tensor_augmented = None

    def forward(self, input_tensor):
        """
        Performs the forward pass of the fully connected layer.
        input_tensor is a matrix with input_size columns and batch_size rows.
        Output Y_hat_prime = X_prime @ W_prime (using "Our Memory Layout" from slides)

        Args:
            input_tensor (np.ndarray): The input data, shape (batch_size, input_size).
        
        Returns:
            np.ndarray: The output of the layer, shape (batch_size, output_size).
        """
        batch_size = input_tensor.shape[0]

        # Augment the input tensor to include the bias term (a column of ones).
        ones_column = np.ones((batch_size, 1))
        self._input_tensor_augmented = np.concatenate((input_tensor, ones_column), axis=1)
        
        # Perform the matrix multiplication: Y_hat_prime = X_prime @ W_prime
        output_tensor = np.dot(self._input_tensor_augmented, self.weights)
        
        return output_tensor

    def backward(self, error_tensor):
        """
        Performs the backward pass of the fully connected layer.
        error_tensor (E_n_prime) is the gradient of the loss with respect to the output of this layer.

        Args:
            error_tensor (np.ndarray): Gradient from the next layer, shape (batch_size, output_size).
        
        Returns:
            np.ndarray: Gradient of the loss w.r.t. the input of this layer (E_n-1_prime, without bias part),
                        shape (batch_size, input_size).
        """
        # 1. Calculate the gradient of the loss with respect to the weights (Delta W_prime).
        self._gradient_weights = np.dot(self._input_tensor_augmented.T, error_tensor)
        
        # 2. Update weights if the layer is trainable and an optimizer is set.
        if self.trainable and self.optimizer is not None:
            update_value = self.optimizer.calculate_update(self.weights, self._gradient_weights)
            self.weights -= update_value
        
        # 3. Calculate the gradient of the loss with respect to the augmented input (E_n-1_prime).
        grad_input_augmented = np.dot(error_tensor, self.weights.T)
        
        # 4. The gradient passed to the previous layer should not include the part for the bias.
        grad_input = grad_input_augmented[:, :-1]
        
        return grad_input
        
    @property
    def gradient_weights(self):
        """
        Returns the gradient with respect to the weights, calculated in the last backward pass.
        
        Returns:
            np.ndarray or None: The gradient of the weights, shape (input_size + 1, output_size).
                                Returns None if backward pass has not been called yet.
        """
        return self._gradient_weights
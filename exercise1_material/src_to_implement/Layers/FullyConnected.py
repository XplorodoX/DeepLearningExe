import numpy as np

class FullyConnected:
    def __init__(self, input_size, output_size):
        """
        Initializes a fully connected layer.

        Args:
            input_size (int): The number of features in the input.
            output_size (int): The number of features in the output.
        """
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = np.random.randn(input_size + 1, output_size) * 0.01
        
        self.trainable = True
        self.optimizer = None  
        self._learning_rate = 0.01 


        self.input_tensor_augmented = None 
        self.original_input_ndim = None

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
        # This creates X_prime.
        # input_tensor: (batch_size, input_size)
        # ones_column: (batch_size, 1)
        # self._input_tensor_augmented: (batch_size, input_size + 1)
        ones_column = np.ones((batch_size, 1))
        self._input_tensor_augmented = np.concatenate((input_tensor, ones_column), axis=1)
        
        # Perform the matrix multiplication: Y_hat_prime = X_prime @ W_prime
        # self._input_tensor_augmented: (batch_size, input_size + 1)
        # self.weights: (input_size + 1, output_size)
        # output_tensor: (batch_size, output_size)
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
        # This is grad_W' = X_prime.T @ E_n_prime (Equation 11, page 31 of slides)
        # self._input_tensor_augmented.T: (input_size + 1, batch_size)
        # error_tensor (E_n_prime): (batch_size, output_size)
        # self._gradient_weights: (input_size + 1, output_size)
        self._gradient_weights = np.dot(self._input_tensor_augmented.T, error_tensor)
        
        # 2. Update weights if the layer is trainable and an optimizer is set.
        if self.trainable and self.optimizer is not None:
            # The optimizer's calculate_update method should return the value to be subtracted
            # (e.g., learning_rate * gradient_tensor).
            update_value = self.optimizer.calculate_update(self.weights, self._gradient_weights)
            self.weights -= update_value
        
        # 3. Calculate the gradient of the loss with respect to the augmented input (E_n-1_prime).
        # This is E_n-1_prime = E_n_prime @ W_prime.T (Equation 10, page 31 of slides)
        # error_tensor (E_n_prime): (batch_size, output_size)
        # self.weights.T: (output_size, input_size + 1)
        # grad_input_augmented: (batch_size, input_size + 1)
        grad_input_augmented = np.dot(error_tensor, self.weights.T)
        
        # 4. The gradient passed to the previous layer should not include the part for the bias.
        # So, remove the last column (gradient w.r.t. the column of ones).
        # grad_input: (batch_size, input_size)
        grad_input = grad_input_augmented[:, :-1]
        
        return grad_input

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Calculates the update for weights using an optimizer or basic SGD.
        This is based on the update rule W_new = W_old - learning_rate * gradient [cite: 44, 47]
        (where eta on slide 31 is the learning rate).

        Args:
            weight_tensor (np.ndarray): The current weights.
            gradient_tensor (np.ndarray): The gradient of the loss w.r.t. the weights.

        Returns:
            np.ndarray: The calculated update value for the weights.
        """
        if not self.trainable:
            return np.zeros_like(gradient_tensor)
            
        if self.optimizer:
            # Assume the optimizer has a method 'calculate_update'
            return self.optimizer.calculate_update(weight_tensor, gradient_tensor)
        else:
            if self._learning_rate is None:
                raise ValueError("Learning rate is not set and no optimizer is provided.")
            return self._learning_rate * gradient_tensor
        
    @property
    def gradient_weights(self):
        """
        Returns the gradient with respect to the weights, calculated in the last backward pass.
        
        Returns:
            np.ndarray or None: The gradient of the weights, shape (input_size + 1, output_size).
                                Returns None if backward pass has not been called yet.
        """
        return self._gradient_weights
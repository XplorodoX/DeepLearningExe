import numpy as np
import copy # Import the copy module

class NeuralNetwork:
    def __init__(self, optimizer):
        """
        Initialize the Neural Network.
        :param optimizer: Optimizer instance to be used for weight updates in layers.
        """
        self.optimizer = optimizer
        self.layers = []
        self.loss = []  # Task refers to this as 'loss' list
        self.data_layer = None  # Will be set by unit tests
        self.loss_layer = None  # Will be set by unit tests (replaces loss_function)

    def forward(self, input_tensor=None):
        """
        Perform a forward pass through the network's layers.
        If input_tensor is provided, use it directly.
        Otherwise, get data from self.data_layer.
        :param input_tensor: Optional input tensor. If None, data is fetched from data_layer.
        :return: Output tensor from the last layer in self.layers (predictions).
        """
        if input_tensor is None:
            # Use data_layer when no input_tensor is provided
            if self.data_layer is None:
                raise ValueError("Data layer not set for the network. Cannot perform forward pass.")
            input_tensor, _ = self.data_layer.next()
    
        current_output = input_tensor
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def backward(self, label_tensor):
        """
        Perform a backward pass through the network.
        This starts with the gradient from the loss_layer.
        :param label_tensor: True labels.
        """
        if self.loss_layer is None:
            raise ValueError("Loss layer not set for the network. Cannot perform backward pass.")

        # Calculate the gradient of the loss with respect to the network's output
        error_tensor = self.loss_layer.backward(label_tensor)

        # Propagate the error backward through the layers in reverse order
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        """
        Append a new layer to the network.
        If the layer is trainable and the network has an optimizer,
        a deep copy of the network's optimizer is assigned to the layer.
        :param layer: Layer to be added (must be an instance of a BaseLayer subclass).
        """
        # Assign a deep copy of the network's optimizer to trainable layers
        if hasattr(layer, 'trainable') and layer.trainable and self.optimizer is not None:
            layer.optimizer = copy.deepcopy(self.optimizer)
    
        self.layers.append(layer)

    def train(self, iterations):
        """
        Train the network for a specified number of iterations.
        Data is fetched from self.data_layer for each iteration.
        Loss is calculated using self.loss_layer.
        :param iterations: Number of training iterations.
        """
        if self.data_layer is None:
            raise ValueError("Data layer not set for the network. Cannot train.")
        if self.loss_layer is None:
            raise ValueError("Loss layer not set for the network. Cannot train.")

        self.loss = []  # Clear loss history for the new training session
        for i in range(iterations):
            # 1. Get input and labels from the data layer for the current iteration/batch
            input_tensor, label_tensor = self.data_layer.next()

            # 2. Forward pass: Get predictions from the network
            predictions = self.forward(input_tensor)

            # 3. Calculate loss using the loss_layer
            current_loss = self.loss_layer.forward(predictions, label_tensor)
            self.loss.append(current_loss)

            # 4. Backward pass: Calculate gradients and update weights
            self.backward(label_tensor)

            # Print progress at intervals
            if (i + 1) % max(1, iterations // 10) == 0 or i == iterations - 1:
                print(f"Iteration {i + 1}/{iterations}, Loss: {current_loss:.4f}")

    def test(self, input_tensor):
        """
        Test the network on a given input_tensor.
        Propagates the input_tensor through the network and returns the prediction
        of the last layer in self.layers. No loss is calculated here, and no
        backward pass or weight updates are performed.
        :param input_tensor: Input tensor for testing.
        :return: Prediction tensor from the last layer of the network.
        """
        # Perform a forward pass to get predictions
        predictions = self.forward(input_tensor)
        return predictions


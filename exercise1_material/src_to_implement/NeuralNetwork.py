import numpy as np
import copy # Import the copy module

class NeuralNetwork:
    def __init__(self, optimizer):
        """
        Initialize the Neural Network.
        :param optimizer: Optimizer instance to be used for weight updates in layers.
        # The loss_function parameter was removed from __init__ as per the latest snippet,
        # assuming it's set elsewhere or not part of this specific problem.
        # If it's needed, it should be added back.
        """
        self.optimizer = optimizer
        self.layers = []
        self.loss_history = []
        self.loss_function = None # Initialize loss_function attribute

    def forward(self, input_tensor):
        """
        Perform a forward pass through the network.
        :param input_tensor: Input tensor to the network.
        :return: Output tensor from the network (predictions).
        """
        current_output = input_tensor
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def backward(self, label_tensor, prediction_tensor):
        """
        Perform a backward pass through the network.
        This starts with the gradient from the loss function.
        :param label_tensor: True labels.
        :param prediction_tensor: Predictions from the forward pass.
        """
        if self.loss_function is None:
            raise ValueError("Loss function not set for the network.")

        # Calculate the gradient of the loss with respect to the network's output
        error_tensor = self.loss_function.backward(prediction_tensor, label_tensor)

        # Propagate the error backward through the layers
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
            # Note: Weight updates are handled within each layer's backward method
            # if its optimizer is set and it's trainable, as per FullyConnected design.

    def append_layer(self, layer):
        """
        Append a new layer to the network.
        The layer will be assigned a deep copy of the network's optimizer instance if trainable.
        :param layer: Layer to be added (must be an instance of a BaseLayer subclass).
        """
        if layer.trainable and self.optimizer is not None:
            # Assign a deep copy of the network's optimizer to the layer
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, input_tensor, label_tensor, iterations):
        """
        Train the network for a specified number of iterations.
        :param input_tensor: Training input data.
        :param label_tensor: Training true labels.
        :param iterations: Number of training iterations.
        """
        if self.loss_function is None:
            raise ValueError("Loss function not set for the network. Cannot train.")

        self.loss_history = []
        for i in range(iterations):
            # 1. Forward pass
            predictions = self.forward(input_tensor)

            # 2. Calculate loss
            current_loss = self.loss_function.forward(predictions, label_tensor)
            self.loss_history.append(current_loss)

            # 3. Backward pass (calculates gradients and updates weights in layers)
            self.backward(label_tensor, predictions)

            if (i + 1) % max(1, iterations // 10) == 0 or i == iterations -1 : # Print progress
                print(f"Iteration {i+1}/{iterations}, Loss: {current_loss:.4f}")


    def test(self, input_tensor, label_tensor):
        """
        Test the network on a given input and labels.
        Calculates the loss without performing a backward pass or updating weights.
        :param input_tensor: Input tensor for testing.
        :param label_tensor: True labels for the input tensor.
        :return: Loss value for the test data.
        """
        if self.loss_function is None:
            raise ValueError("Loss function not set for the network. Cannot test.")

        predictions = self.forward(input_tensor)
        loss = self.loss_function.forward(predictions, label_tensor)
        return loss
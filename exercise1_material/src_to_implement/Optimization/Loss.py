import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        """
        Initialize the CrossEntropyLoss.
        """
        self.predictions = None
        self.targets = None

    def forward(self, prediction_tensor, label_tensor):
        """
        Compute the forward pass of the loss function.
        :param prediction_tensor: Predicted probabilities (output of the model).
        :param label_tensor: True labels (one-hot encoded).
        :return: Computed loss value.
        """
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-15  # More standard value
        # Ensure prediction values are positive (between epsilon and 1)
        prediction_tensor = np.clip(prediction_tensor, epsilon, 1 - epsilon)
        
        # Store tensors for backward pass
        self.predictions = prediction_tensor
        self.targets = label_tensor
        
        # Calculate -y*log(y')
        loss = -np.sum(label_tensor * np.log(prediction_tensor))
        
        return loss

    def backward(self, label_tensor=None):
        """
        Compute the backward pass of the loss function.
        :param label_tensor: Optional label tensor (ignored, using stored targets)
        :return: Gradient of the loss with respect to predictions.
        """
        # Gradient is -y/y'
        grad = -self.targets / self.predictions
        return grad
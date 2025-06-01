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
       
        
        eps = np.finfo(float).eps
        
        # Add small epsilon to avoid log(0)
        prediction_tensor_safe = np.maximum(prediction_tensor, eps)

        # Store tensors for backward pass
        self.predictions = prediction_tensor_safe
        self.targets = label_tensor
        
        # Cross entropy loss: -Σ(y_true * log(y_pred))
        # Sum across all elements, not normalized by batch size
        loss = -np.sum(label_tensor * np.log(prediction_tensor_safe))
        
        return loss

    def backward(self, label_tensor=None):
        """
        Compute the backward pass of the loss function.
        :param label_tensor: Optional label tensor (ignored, using stored targets)
        :return: Gradient of the loss with respect to predictions.
        """
        # If label_tensor is provided, use it instead of stored targets
        if label_tensor is not None:
            self.targets = label_tensor
        
        # Gradient of cross entropy with respect to predictions is -y_true/y_pred
        gradient = -self.targets / self.predictions + np.finfo(float).eps
        
        return gradient
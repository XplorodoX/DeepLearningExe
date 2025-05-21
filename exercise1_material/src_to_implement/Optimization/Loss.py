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
        :param predictions: Predicted probabilities (output of the model).
        :param targets: True labels (ground truth).
        :return: Computed loss value.
        """
        self.predictions = predictions
        self.targets = label_tensor

        # Clip predictions to avoid log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)

        # Compute cross-entropy loss
        loss = -np.sum(label_tensor * np.log(predictions)) / label_tensor.shape[0]
        return loss
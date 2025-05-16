import numpy as np

class FullyConnected:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, (input_size, output_size))
        self.bias = None
        self.trainable = True


    def forward(self, input_tensor):
        # Implement the forward pass
        pass

    def backward(self, grad_output):
        # Implement the backward pass
        pass

    def update(self, learning_rate):
        # Implement the weights and bias update
        pass
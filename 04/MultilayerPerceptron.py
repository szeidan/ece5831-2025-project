import numpy as np

class MultilayerPerceptron:
    def __init__(self):
        # Initialize weights and biases for each layer
        self.w1 = np.array([[1, 3, 5], [2, 4, 6]])
        self.b1 = np.array([0.1, 0.2, 0.3])

        self.w2 = np.array([[1, 4], [2, 5], [3, 6]])
        self.b2 = np.array([0.4, 0.3])

        self.w3 = np.array([[1, 3],[2, 4]])
        self.b3 = np.array([0.9, 0.2])

    def identity(self, x):
        return x
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        # Forward pass through the network
        a1 = np.dot(x, self.w1) + self.b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1, self.w2) + self.b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2, self.w3) + self.b3
        y = self.identity(a3) # No activation function in output layer
        return y


if __name__ == "__main__":
    print("MultilayerPerceptron Example by Sobhi Zeidan")
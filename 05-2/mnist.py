
import os
import pickle
from typing import Tuple

import numpy as np

from mnist_data import stable_softmax

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

class Mnist:
    """
    Simple 3‑layer MLP inference for MNIST using pretrained weights (sample_weight.pkl).
    Expected weight dict keys: W1, b1, W2, b2, W3, b3
    """
    def __init__(self, model_path: str = "model/sample_weight.pkl"):
        self.model_path = model_path
        self.network = self._init_network()

    def _init_network(self) -> dict:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Pretrained weights not found at {self.model_path}. "
                "Place sample_weight.pkl under model/ or pass --model."
            )
        with open(self.model_path, "rb") as f:
            net = pickle.load(f)
        return net

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        x: (N,784) or (784,)
        returns: softmax probabilities (N,10) or (10,)
        """
        single = False
        if x.ndim == 1:
            x = x[None, :]
            single = True

        W1, W2, W3 = self.network["W1"], self.network["W2"], self.network["W3"]
        b1, b2, b3 = self.network["b1"], self.network["b2"], self.network["b3"]

        a1 = x @ W1 + b1
        z1 = sigmoid(a1)
        a2 = z1 @ W2 + b2
        z2 = sigmoid(a2)
        a3 = z2 @ W3 + b3
        y  = stable_softmax(a3)

        if single:
            return y[0]
        return y
    def accuracy(self, X: np.ndarray, y: np.ndarray, batch_size: int = 100) -> float:
        """
        Compute accuracy of the network on (X, y).
        Parameters
        ----------
        X : np.ndarray
            Test images (N, 784)
        y : np.ndarray
            True labels (N,)
        batch_size : int
            How many samples to predict at once.
        Returns
        -------
        float
            Accuracy = correct_predictions / total_samples
        """
        n = X.shape[0]
        correct = 0
        for i in range(0, n, batch_size):
            xb = X[i:i + batch_size]
            yb = y[i:i + batch_size]
            probs = self.predict(xb)
            pred = np.argmax(probs, axis=1)
            correct += np.sum(pred == yb)
        return correct / n

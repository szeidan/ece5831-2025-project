# layer.py
# -----------------------------------------------------------------------------
# Minimal set of layers needed for a two-layer MLP with backpropagation:
#   - Affine (Fully-Connected) layer
#   - ReLU activation
#   - SoftmaxWithLoss (combines softmax + cross-entropy loss)
# Each layer implements forward() and backward() where applicable.
# -----------------------------------------------------------------------------

import numpy as np


class Affine:
    """
    Fully-connected (dense) layer: y = xW + b
    - Stores input x during forward pass for use in backward pass.
    - During backward, computes gradients for W, b, and input x.
    """
    def __init__(self, W, b):
        """
        Args:
            W (np.ndarray): Weight matrix of shape (D, M)
            b (np.ndarray): Bias vector of shape (M,)
        """
        self.W = W
        self.b = b
        self.x = None      # cached input
        self.dW = None     # gradient wrt W
        self.db = None     # gradient wrt b

    def forward(self, x):
        """
        Args:
            x (np.ndarray): Input of shape (N, D)
        Returns:
            np.ndarray: Output of shape (N, M)
        """
        self.x = x
        out = x @ self.W + self.b  # matrix multiply + bias
        return out

    def backward(self, dout):
        """
        Backprop through affine transform.
        Args:
            dout (np.ndarray): Upstream gradient of shape (N, M)
        Returns:
            np.ndarray: Gradient wrt input x, shape (N, D)
        """
        # dW = x^T * dout
        self.dW = self.x.T @ dout
        # db = sum over batch for each output dim
        self.db = np.sum(dout, axis=0)
        # dx = dout * W^T
        dx = dout @ self.W.T
        return dx


class ReLU:
    """
    Rectified Linear Unit activation:
        ReLU(x) = max(0, x)
    Stores mask of where x <= 0 to zero-out gradients in backward pass.
    """
    def __init__(self):
        self.mask = None

    def forward(self, x):
        """
        Args:
            x (np.ndarray): Input of any shape
        Returns:
            np.ndarray: Same shape as x with negative values clamped to 0
        """
        self.mask = (x <= 0)  # boolean mask of non-positive entries
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        """
        Backprop through ReLU:
        - Gradient is blocked (set to 0) where input was <= 0.
        Args:
            dout (np.ndarray): Upstream gradient (same shape as input)
        Returns:
            np.ndarray: Gradient wrt input
        """
        dout[self.mask] = 0
        return dout


class SoftmaxWithLoss:
    """
    Softmax activation combined with cross-entropy loss.
    - forward(x, t) returns scalar loss
    - backward() returns gradient wrt x
    Assumes t is one-hot encoded.
    """
    def __init__(self):
        self.y = None     # softmax output (N, C)
        self.t = None     # ground truth one-hot (N, C)
        self.loss_val = None

    @staticmethod
    def _softmax(z):
        """
        Numerically stable softmax along classes (axis=1).
        """
        z = z - np.max(z, axis=1, keepdims=True)  # stabilize
        expz = np.exp(z)
        return expz / np.sum(expz, axis=1, keepdims=True)

    def forward(self, x, t):
        """
        Args:
            x (np.ndarray): Raw class scores (N, C)
            t (np.ndarray): One-hot ground truth (N, C)
        Returns:
            float: Cross-entropy loss
        """
        self.t = t
        self.y = self._softmax(x)
        eps = 1e-12
        logp = np.log(self.y + eps)  # avoid log(0)
        loss = -np.sum(t * logp) / x.shape[0]
        self.loss_val = loss
        return loss

    def backward(self, dout=1.0):
        """
        Gradient wrt input scores x for softmax + cross-entropy:
            dL/dx = (softmax(x) - t) / N
        Args:
            dout (float): Upstream scalar multiplier (rarely used; keep as 1)
        Returns:
            np.ndarray: Gradient wrt x (N, C)
        """
        N = self.t.shape[0]
        dx = (self.y - self.t) / N
        return dx * dout

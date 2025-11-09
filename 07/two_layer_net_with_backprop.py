# two_layer_net_with_backprop.py
# -----------------------------------------------------------------------------
# Two-layer neural network (MLP) with explicit backprop:
#   input(784) -> Affine(W1,b1) -> ReLU -> Affine(W2,b2) -> SoftmaxWithLoss
# Provides:
#   - predict(x): forward pass producing logits
#   - loss(x,t): softmax cross-entropy + optional L2 regularization
#   - gradient(x,t): backprop to compute parameter gradients
#   - accuracy(x,t_onehot): convenience metric
# -----------------------------------------------------------------------------

import numpy as np
from layer import Affine, ReLU, SoftmaxWithLoss


class TwoLayerNetWithBackprop:
    """
    Simple 2-layer MLP: input -> Affine -> ReLU -> Affine -> Softmax
    """
    def __init__(
        self,
        input_size=784,
        hidden_size=100,
        output_size=10,
        weight_init_std=0.01,
        reg=0.0,
        seed=42
    ):
        # Random generator for reproducibility
        rng = np.random.default_rng(seed)

        # Parameter dictionaries (weights and biases)
        self.params = {}
        self.params["W1"] = weight_init_std * rng.standard_normal((input_size, hidden_size)).astype(np.float32)
        self.params["b1"] = np.zeros(hidden_size, dtype=np.float32)
        self.params["W2"] = weight_init_std * rng.standard_normal((hidden_size, output_size)).astype(np.float32)
        self.params["b2"] = np.zeros(output_size, dtype=np.float32)

        # L2 regularization strength
        self.reg = reg

        # Compose network from layers.
        # NOTE: Affine layers store direct references to W and b, so if we
        #       replace self.params[...] we must also update the layer refs.
        self.affine1 = Affine(self.params["W1"], self.params["b1"])
        self.relu = ReLU()
        self.affine2 = Affine(self.params["W2"], self.params["b2"])
        self.softmax_loss = SoftmaxWithLoss()

    # ---------------------- forward path helpers ----------------------
    def predict(self, x):
        """
        Forward pass producing raw logits (no softmax applied here).
        Args:
            x (np.ndarray): Input batch (N, 784) normalized to [0,1].
        Returns:
            np.ndarray: Logits (N, 10).
        """
        out = self.affine1.forward(x)  # (N, hidden)
        out = self.relu.forward(out)   # ReLU
        out = self.affine2.forward(out)  # (N, 10) logits
        return out

    def loss(self, x, t):
        """
        Compute softmax cross-entropy loss + L2 regularization term.
        Args:
            x (np.ndarray): Input batch (N, 784)
            t (np.ndarray): One-hot labels (N, 10)
        Returns:
            float: Scalar loss
        """
        scores = self.predict(x)
        data_loss = self.softmax_loss.forward(scores, t)
        reg_loss = 0.5 * self.reg * (np.sum(self.params["W1"]**2) + np.sum(self.params["W2"]**2))
        return data_loss + reg_loss

    def accuracy(self, x, t_onehot):
        """
        Convenience method to compute accuracy.
        """
        logits = self.predict(x)
        pred = np.argmax(logits, axis=1)
        true = np.argmax(t_onehot, axis=1)
        return float(np.mean(pred == true))

    # ---------------------- backward (gradients) ----------------------
    def gradient(self, x, t):
        """
        Run backpropagation to compute gradients for parameters.
        Typical usage: call loss() first (for consistent caches), but we
        re-run forward here as well for safety.
        Returns:
            dict[str, np.ndarray]: grads for W1,b1,W2,b2 (with L2 terms for W1,W2)
        """
        # Forward (ensures softmax_loss has current caches for backward)
        scores = self.predict(x)
        self.softmax_loss.forward(scores, t)

        # Backward pass (reverse order of forward):
        # dL/dscores
        dout = self.softmax_loss.backward()
        # through final affine
        dout = self.affine2.backward(dout)
        # through ReLU
        dout = self.relu.backward(dout)
        # through first affine
        _ = self.affine1.backward(dout)

        # Collect grads; add L2 weight decay to W1, W2
        grads = {
            "W1": self.affine1.dW + self.reg * self.params["W1"],
            "b1": self.affine1.db,
            "W2": self.affine2.dW + self.reg * self.params["W2"],
            "b2": self.affine2.db,
        }
        return grads

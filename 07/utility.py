# utility.py
# -----------------------------------------------------------------------------
# General helper utilities used across the TwoLayerNet project.
# Contains one-hot encoding, accuracy calculation, and (de)serialization helpers.
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pickle


def one_hot(y, num_classes):
    """
    Convert integer labels into one-hot encoded rows.
    Args:
        y (np.ndarray): Shape (N,) integer labels.
        num_classes (int): Number of classes.
    Returns:
        np.ndarray: Shape (N, num_classes) float32 one-hot matrix.
    """
    y = y.astype(int).ravel()
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def accuracy(y_pred_logits, y_true_onehot):
    """
    Compute classification accuracy.
    Args:
        y_pred_logits (np.ndarray): Raw class scores (N, C).
        y_true_onehot (np.ndarray): One-hot ground-truth (N, C).
    Returns:
        float: Accuracy in [0,1].
    """
    y_pred = np.argmax(y_pred_logits, axis=1)
    y_true = np.argmax(y_true_onehot, axis=1)
    return float(np.mean(y_pred == y_true))


def save_pickle(obj, filename: str):
    """
    Serialize a Python object to disk using pickle.
    Creates parent directories if needed.
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename: str):
    """Load and return a Python object from a pickle file."""
    with open(filename, "rb") as f:
        return pickle.load(f)

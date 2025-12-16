# mnist_data.py
# -----------------------------------------------------------------------------
# Loads MNIST and returns:
#   (x_train, y_train_oh), (x_test, y_test_oh)
# - Images are flattened to 784-dim and normalized to [0,1].
# - Labels are converted to one-hot representation.
# If tensorflow.keras.datasets.mnist is unavailable, tries fetch_openml (sklearn).
# -----------------------------------------------------------------------------

import numpy as np


def _one_hot(y, num_classes):
    """Local one-hot helper to avoid import cycles."""
    y = y.astype(int).ravel()
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def load_mnist(normalize=True, dtype=np.float32, seed=0):
    """
    Returns:
        (x_train, y_train_oh), (x_test, y_test_oh)
        x_*: (N, 784) float32 in [0,1] if normalize else [0,255]
        y_*_oh: (N, 10) one-hot
    Notes:
        - Prefers tensorflow.keras.datasets.mnist for simplicity/reliability.
        - Fallback: sklearn's fetch_openml (requires internet access).
    """
    try:
        # Preferred source (bundled with TF; downloads once and caches)
        from tensorflow.keras.datasets import mnist  # type: ignore
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    except Exception:
        # Optional fallback via scikit-learn
        try:
            from sklearn.datasets import fetch_openml  # type: ignore
            data = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
            X = data['data'].astype(dtype)
            y = data['target'].astype(np.int64)
            x_train, x_test = X[:60000], X[60000:]
            y_train, y_test = y[:60000], y[60000:]
            x_train = x_train.reshape(-1, 28, 28)
            x_test = x_test.reshape(-1, 28, 28)
        except Exception as e2:
            raise RuntimeError(
                "Could not load MNIST via tensorflow.keras OR sklearn.\n"
                "Install TensorFlow (recommended) or scikit-learn, or place your own loader."
            ) from e2

    # Flatten to 784 and cast
    x_train = x_train.reshape((-1, 784)).astype(dtype)
    x_test  = x_test.reshape((-1, 784)).astype(dtype)

    # Normalize if requested
    if normalize:
        x_train /= 255.0
        x_test  /= 255.0

    # One-hot labels
    y_train_oh = _one_hot(y_train, 10)
    y_test_oh  = _one_hot(y_test, 10)
    return (x_train, y_train_oh), (x_test, y_test_oh)

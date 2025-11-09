# train.py
# -----------------------------------------------------------------------------
# Trains a two-layer neural network on MNIST and saves learned parameters
# to a pickle file named after the last name requirement:
#   "<lastname>_mnist_model.pkl" (here, "zeidan_mnist_model.pkl")
# Hyperparameters are intentionally simple and easy to tweak.
# -----------------------------------------------------------------------------

import numpy as np
from two_layer_net_with_backprop import TwoLayerNetWithBackprop
from mnist_data import load_mnist
from utility import accuracy, save_pickle

# ====== EDIT THIS to your last name if needed (assignment rule) ======
LAST_NAME = "zeidan"
MODEL_PATH = f"{LAST_NAME}_mnist_model.pkl"
# =====================================================================


def sgd(param, grad, lr):
    """
    In-place Stochastic Gradient Descent update:
        param <- param - lr * grad
    """
    param -= lr * grad


def iterate_minibatches(X, Y, batch_size, shuffle=True, rng=None):
    """
    Yield mini-batches (Xb, Yb) of size batch_size.
    Args:
        X (np.ndarray): Inputs (N, D)
        Y (np.ndarray): One-hot labels (N, C)
    """
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        if rng is None:
            rng = np.random.default_rng(123)
        rng.shuffle(idx)
    for start in range(0, N, batch_size):
        end = start + batch_size
        batch_idx = idx[start:end]
        yield X[batch_idx], Y[batch_idx]


def main():
    # ---------------- Load data ----------------
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=True)

    # ---------------- Build network ----------------
    net = TwoLayerNetWithBackprop(
        input_size=784,
        hidden_size=128,   # try 256 for better accuracy (slower)
        output_size=10,
        weight_init_std=0.01,
        reg=1e-4,          # small L2 regularization helps generalization
        seed=7
    )

    # ---------------- Training setup ----------------
    lr = 0.1         # learning rate
    batch_size = 128
    epochs = 5       # 10-15 can reach ~97-98% with good tuning
    print_every = 100

    itr = 0
    for ep in range(1, epochs + 1):
        for xb, yb in iterate_minibatches(x_train, y_train, batch_size, shuffle=True):
            # Forward pass + loss
            loss = net.loss(xb, yb)

            # Backpropagate to compute gradients
            grads = net.gradient(xb, yb)

            # Parameter updates (SGD)
            for k in ["W1", "b1", "W2", "b2"]:
                sgd(net.params[k], grads[k], lr)

            # Optional training progress print
            if (itr % print_every) == 0:
                train_acc = net.accuracy(xb, yb)
                print(f"epoch {ep}/{epochs} | iter {itr:05d} | loss {loss:.4f} | batch_acc {train_acc:.3f}")
            itr += 1

        # Evaluate on test set each epoch
        test_logits = net.predict(x_test)
        test_acc = accuracy(test_logits, y_test)
        print(f"[Epoch {ep}] Test accuracy: {test_acc:.4f}")

    # ---------------- Save model ----------------
    to_save = {
        "params": net.params,      # learned weights/biases
        "meta": {                  # useful metadata to reconstruct the net later
            "input_size": 784, "hidden_size": 128, "output_size": 10,
            "reg": net.reg, "epochs": epochs, "batch_size": batch_size, "lr": lr
        }
    }
    save_pickle(to_save, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

"""
thresholds.py
-------------
This module computes an optimal classification threshold using the
validation set. The default metric is the F1 score. A better threshold
often improves fault detection by balancing precision and recall.

Run:
    python -m src.thresholds --ckpt runs/cnn1d_run/best.pt
"""

import argparse
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from . import config
from .models import CNN1D, LSTMseq, TCN


def load_model(ckpt_path):
    """Loads a saved PyTorch model from checkpoint."""
    ck = torch.load(ckpt_path, map_location="cpu")
    name, in_ch = ck["model_name"], ck["in_ch"]

    Model = {
        "cnn1d": CNN1D,
        "lstm": LSTMseq,
        "tcn": TCN
    }[name]

    model = Model(in_ch=in_ch, n_classes=2)
    model.load_state_dict(ck["model"])
    model.eval()

    return model


def compute_best_threshold(model, Xv, yv):
    """
    Computes the threshold that maximizes F1 score on the validation set.
    """
    with torch.no_grad():
        logits = model(torch.tensor(Xv, dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()

    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(yv, probs)

    # Compute F1 (avoid divide-by-zero)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    best_idx = f1.argmax()
    best_threshold = thresholds[best_idx]
    best_f1 = f1[best_idx]

    return best_threshold, best_f1, precision, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    # Load validation windows
    Xv = np.load(config.WINDOWS_DIR / "X_val.npy")
    yv = np.load(config.WINDOWS_DIR / "y_val.npy")

    # Load model
    model = load_model(args.ckpt)

    # Compute threshold
    th, f1, precision, recall = compute_best_threshold(model, Xv, yv)

    print(f"\n[Threshold Tuning]")
    print(f"Best threshold: {th:.4f}")
    print(f"Best F1 score: {f1:.4f}")

    # Save threshold
    out_file = config.RUNS_DIR / "best_threshold.txt"
    with open(out_file, "w") as f:
        f.write(f"{th:.6f}\n")

    print(f"Saved threshold to: {out_file}")


if __name__ == "__main__":
    main()

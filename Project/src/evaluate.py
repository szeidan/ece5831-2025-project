"""
evaluate.py
------------
Evaluate a trained checkpoint on the test set. Writes PR/ROC curves and metrics.
Run: python -m src.evaluate --ckpt runs/cnn1d_run/best.pt --out runs/cnn1d_run
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from . import config
from .models import CNN1D, LSTMseq, TCN

def load_model(ckpt_path: Path, in_ch_override=None):
    ck = torch.load(ckpt_path, map_location="cpu")
    name = ck["model_name"]
    in_ch = ck["in_ch"] if in_ch_override is None else in_ch_override
    if name == "cnn1d":
        m = CNN1D(in_ch=in_ch, n_classes=2)
    elif name == "lstm":
        m = LSTMseq(in_ch=in_ch, n_classes=2)
    else:
        m = TCN(in_ch=in_ch, n_classes=2)
    m.load_state_dict(ck["model"]); m.eval()
    return name, m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", type=str, default=str((config.RUNS_DIR / "eval").as_posix()))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    Xte = np.load(config.WINDOWS_DIR / "X_test.npy")
    yte = np.load(config.WINDOWS_DIR / "y_test.npy")
    in_ch = Xte.shape[-1]

    name, model = load_model(Path(args.ckpt), in_ch_override=in_ch)

    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(Xte, dtype=torch.float32)), dim=-1)[:, 1].numpy()
    ypred = (probs >= 0.5).astype(int)

    # Curves
    p, r, _ = precision_recall_curve(yte, probs)
    ap_score = auc(r, p)
    fpr, tpr, _ = roc_curve(yte, probs)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(yte, ypred)

    # Plots
    plt.figure()
    plt.plot(r, p)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curve (AP={ap_score:.3f})")
    plt.savefig(out_dir / "pr_curve.png", dpi=150)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC Curve (AUC={roc_auc:.3f})")
    plt.savefig(out_dir / "roc_curve.png", dpi=150)

    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"AP={ap_score:.4f} ROC_AUC={roc_auc:.4f}\n")
        f.write("Confusion matrix (threshold=0.5):\n")
        f.write(str(cm) + "\n\n")
        f.write(str(classification_report(yte, ypred, digits=3)) + "\n")

    print("Wrote plots & metrics to", out_dir.as_posix())

if __name__ == "__main__":
    main()

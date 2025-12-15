"""
train.py
---------
Train a model on the windowed dataset.
Run: python -m src.train --model cnn1d --epochs 10 --out runs/cnn1d_run
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tqdm import tqdm
from . import config
from .models import CNN1D, LSTMseq, TCN

def get_model(name: str, in_ch: int, n_classes: int = 2):
    if name == "cnn1d":
        return CNN1D(in_ch=in_ch, n_classes=n_classes)
    if name == "lstm":
        return LSTMseq(in_ch=in_ch, n_classes=n_classes)
    if name == "tcn":
        return TCN(in_ch=in_ch, n_classes=n_classes)
    raise ValueError("model must be one of: cnn1d, lstm, tcn")

def load_split():
    Xtr = np.load(config.WINDOWS_DIR / "X_train.npy")
    ytr = np.load(config.WINDOWS_DIR / "y_train.npy")
    Xva = np.load(config.WINDOWS_DIR / "X_val.npy")
    yva = np.load(config.WINDOWS_DIR / "y_val.npy")
    return (Xtr, ytr), (Xva, yva)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn1d", choices=["cnn1d","lstm","tcn"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default=str((config.RUNS_DIR / "run").as_posix()))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    (Xtr, ytr), (Xva, yva) = load_split()
    in_ch = Xtr.shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model, in_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    tr_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
    va_ds = TensorDataset(torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.long))
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False)

    best_ap, best_path = -1.0, out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        tr_loss = 0.0
        for xb, yb in tqdm(tr_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(tr_ds)

        # ---- Validate ----
        model.eval()
        probs, ytrue = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device)
                logits = model(xb)
                p1 = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs.append(p1); ytrue.append(yb.numpy())
        import numpy as np
        probs = np.concatenate(probs); ytrue = np.concatenate(ytrue)
        ap = average_precision_score(ytrue, probs)
        ypred = (probs >= 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(ytrue, ypred, average="binary", zero_division=0)

        print(f"[val] epoch={epoch} loss={tr_loss:.4f} AP={ap:.3f} P={p:.3f} R={r:.3f} F1={f1:.3f}")
        if ap > best_ap:
            best_ap = ap
            torch.save({"model": model.state_dict(), "in_ch": in_ch, "model_name": args.model}, best_path)
            with open(out_dir / "best.txt", "w") as f:
                f.write(f"best AP={best_ap:.4f} epoch={epoch}\n")

    print("Saved best to:", best_path.as_posix())

if __name__ == "__main__":
    main()

# module7.py

import sys
from pathlib import Path
import numpy as np
from PIL import Image

from utility import load_pickle, accuracy
from two_layer_net_with_backprop import TwoLayerNetWithBackprop
from mnist_data import load_mnist

MODEL_FILE = "zeidan_mnist_model.pkl"   # must match train.py


def load_model_as_net(model_path=MODEL_FILE):
    """Reconstruct net and load saved parameters (re-binding layer refs)."""
    data = load_pickle(model_path)
    meta = data["meta"]
    net = TwoLayerNetWithBackprop(
        input_size=meta["input_size"],
        hidden_size=meta["hidden_size"],
        output_size=meta["output_size"],
        reg=meta["reg"],
    )
    for k in net.params:
        net.params[k] = data["params"][k]
    net.affine1.W, net.affine1.b = net.params["W1"], net.params["b1"]
    net.affine2.W, net.affine2.b = net.params["W2"], net.params["b2"]
    return net


def evaluate_on_mnist_test():
    """Evaluate saved model on MNIST test split and print accuracy."""
    (_, _), (x_test, y_test) = load_mnist(normalize=True)
    net = load_model_as_net(MODEL_FILE)
    logits = net.predict(x_test)
    acc = accuracy(logits, y_test)
    print(f"MNIST test accuracy: {acc:.4f}")
    return acc


# ----------------------- Robust MNIST-like preprocessing -----------------------
def _center_of_mass_shift(x_bin):
    """Compute integer shift to move binary mass center to image center (13.5, 13.5)."""
    ys, xs = np.nonzero(x_bin)
    if len(xs) == 0:
        return 0, 0
    cy, cx = ys.mean(), xs.mean()
    dy = int(round(13.5 - cy))
    dx = int(round(13.5 - cx))
    return dy, dx


def _apply_shift(x, dy, dx):
    """Shift image by (dy, dx) with zero-padding (no wrap)."""
    x2 = np.zeros_like(x)
    h, w = x.shape
    ys = slice(max(0, dy), min(h, h + dy))
    xs = slice(max(0, dx), min(w, w + dx))
    ys_from = slice(max(0, -dy), min(h, h - dy))
    xs_from = slice(max(0, -dx), min(w, w - dx))
    x2[ys, xs] = x[ys_from, xs_from]
    return x2


def _prep_own_image(path: Path, export_preview_dir: Path | None = None):
    """
    Convert a custom digit image into MNIST-like input of shape (1, 784).

    Steps:
      - Grayscale
      - Polarity normalization: ensure background is dark, digit bright
      - Contrast stretch to [0, 255]
      - Binarize (threshold)
      - Crop to bounding box
      - Resize longer side to 20 px (keep aspect ratio)
      - Pad to 28x28
      - Center by shifting to center-of-mass
      - Normalize to [0,1]
      - (Optional) export intermediate previews for debugging
    """
    # Load as grayscale float
    img0 = Image.open(path).convert("L")
    arr = np.asarray(img0).astype(np.float32)

    # Heuristic polarity: compare border vs center; if border brighter -> invert
    h, w = arr.shape
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    center = arr[h//4:3*h//4, w//4:3*w//4]
    if border.mean() > center.mean():
        arr = 255.0 - arr  # invert so background tends to dark

    # Contrast stretch to [0, 255]
    arr -= arr.min()
    if arr.max() > 1e-6:
        arr = 255.0 * (arr / arr.max())
    arr_u8 = arr.astype(np.uint8)

    # Binarize and crop to digit bbox
    th = 128
    bin_img = (arr_u8 > th).astype(np.uint8)
    ys, xs = np.where(bin_img == 1)
    if len(xs) == 0 or len(ys) == 0:
        # Fallback: resize entire image
        img28 = Image.fromarray(arr_u8).resize((28, 28), Image.BILINEAR)
        x = np.asarray(img28).astype(np.float32) / 255.0
        if export_preview_dir:
            export_preview_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr_u8).save(export_preview_dir / f"{path.stem}_A_contrast.png")
            img28.save(export_preview_dir / f"{path.stem}_Z_final28.png")
        return x.reshape(1, -1)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cropped = arr_u8[y_min:y_max + 1, x_min:x_max + 1]

    # Resize longest side to 20 px
    ch, cw = cropped.shape
    if ch >= cw:
        new_h, new_w = 20, max(1, int(round(20 * cw / ch)))
    else:
        new_w, new_h = 20, max(1, int(round(20 * ch / cw)))
    resized = Image.fromarray(cropped).resize((new_w, new_h), Image.BILINEAR)

    # Paste into 28x28 canvas (rough center)
    canvas = Image.new("L", (28, 28), color=0)
    top  = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas.paste(resized, (left, top))
    x = np.asarray(canvas).astype(np.float32)

    # CoM center shift
    x_bin = (x > th).astype(np.uint8)
    if x_bin.sum() > 0:
        dy, dx = _center_of_mass_shift(x_bin)
        x = _apply_shift(x, dy, dx)

    # Normalize to [0,1]
    x = x / 255.0

    # Optional preview export
    if export_preview_dir:
        export_preview_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr_u8).save(export_preview_dir / f"{path.stem}_A_contrast.png")
        Image.fromarray(cropped).save(export_preview_dir / f"{path.stem}_B_cropped.png")
        resized.save(export_preview_dir / f"{path.stem}_C_resized20.png")
        Image.fromarray((x * 255).astype(np.uint8)).save(export_preview_dir / f"{path.stem}_Z_final28.png")

    return x.reshape(1, -1)


def test_it_with_your_own_handwritten_digits(image_paths, export_previews=False):
    """
    Classify images and print predictions. If export_previews=True,
    saves intermediate stages to 'preprocessed_preview/'.
    """
    net = load_model_as_net(MODEL_FILE)
    out_dir = Path("preprocessed_preview") if export_previews else None

    for p in image_paths:
        pp = Path(p)
        x = _prep_own_image(pp, export_preview_dir=out_dir)
        logits = net.predict(x)
        pred = int(np.argmax(logits, axis=1)[0])
        print(f"{p}: predicted -> {pred}")


def eval_folder(folder="test_images", pattern="*.png", export_previews=False):
    """
    Convenience: scan a folder and evaluate all images. Prints per-digit accuracy
    if file names are like '7_3.png' where the label is the prefix before '_'.
    """
    net = load_model_as_net(MODEL_FILE)
    folder = Path(folder)
    paths = sorted(folder.glob(pattern))
    if not paths:
        print(f"No images found under {folder}/{pattern}")
        return

    out_dir = Path("preprocessed_preview") if export_previews else None

    totals = {d: 0 for d in range(10)}
    correct = {d: 0 for d in range(10)}

    for p in paths:
        true = None
        try:
            true = int(p.stem.split("_")[0])
        except Exception:
            pass

        x = _prep_own_image(p, export_preview_dir=out_dir)
        pred = int(np.argmax(net.predict(x), axis=1)[0])
        if true is None:
            print(f"{p}: predicted -> {pred}")
        else:
            print(f"{p}: true={true}, pred={pred}")
            totals[true] += 1
            correct[true] += int(pred == true)

    # Report
    has_labels = sum(totals.values()) > 0
    if has_labels:
        print("\nPer-digit accuracy:")
        for d in range(10):
            if totals[d] == 0:
                continue
            acc = correct[d] / totals[d]
            print(f"{d}: {acc:.2f}  ({correct[d]}/{totals[d]})")
        overall = sum(correct.values()) / max(1, sum(totals.values()))
        print(f"\nOverall: {overall:.2f}")


if __name__ == "__main__":
    # 1) Always print MNIST accuracy to confirm the model is reasonable
    evaluate_on_mnist_test()

    # 2) If args given: treat as explicit file list; else scan test_images/*.png
    if len(sys.argv) > 1:
        test_it_with_your_own_handwritten_digits(sys.argv[1:], export_previews=True)
    else:
        eval_folder(folder="test_images", pattern="*.png", export_previews=True)

# module5-3.py
# Test handwritten digit images with pretrained MNIST MLP (sample_weight.pkl)
# Usage examples:
#   python module5-3.py 3_2.png 3
#   python module5-3.py test_images/7_0.png 7
#   python module5-3.py --all

import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from mnist import Mnist

# =============================
# Image Preprocessing Utilities
# =============================

def load_digit_image(path: str, force_resize: bool = True, debug=False) -> np.ndarray:
    """
    Load an image and convert it into MNIST-like format:
      - grayscale
      - auto-invert if background is white
      - crop, scale to 20x20, pad and center in 28x28 frame
      - normalize [0,1]
      - flatten to (784,)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)

    # Auto-invert if background is brighter than center (black ink on white bg)
    h, w = arr.shape
    corners = np.r_[arr[0:4,0:4].ravel(),
                    arr[0:4,-4:].ravel(),
                    arr[-4:,0:4].ravel(),
                    arr[-4:,-4:].ravel()]
    center_patch = arr[h//2-2:h//2+2, w//2-2:w//2+2].ravel()
    if corners.mean() > center_patch.mean():
        arr = 255.0 - arr

    arr = arr / 255.0
    mask = arr > 0.2
    if not mask.any():
        mask = arr > 0.0

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]
    if len(r_idx) == 0 or len(c_idx) == 0:
        return np.zeros((28 * 28,), dtype=np.float32)

    r0, r1 = r_idx[0], r_idx[-1]
    c0, c1 = c_idx[0], c_idx[-1]
    crop = arr[r0:r1+1, c0:c1+1]

    # Scale to 20x20, maintain aspect ratio
    ch, cw = crop.shape
    if ch > cw:
        new_h, new_w = 20, int(round(20 * cw / ch))
    else:
        new_w, new_h = 20, int(round(20 * ch / cw))

    crop_img = Image.fromarray((crop * 255).astype(np.uint8), mode="L")
    crop_resized = crop_img.resize((new_w, new_h), Image.LANCZOS)
    crop_resized = np.asarray(crop_resized, dtype=np.float32) / 255.0

    # Paste centered in 28x28
    canvas = np.zeros((28, 28), dtype=np.float32)
    ro, co = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[ro:ro+new_h, co:co+new_w] = crop_resized

    # Recenter by mass
    ys, xs = np.nonzero(canvas > 0.2)
    if len(ys) > 0:
        values = canvas[ys, xs]
        cy = int(round(np.sum(ys * values) / (np.sum(values) + 1e-8)))
        cx = int(round(np.sum(xs * values) / (np.sum(values) + 1e-8)))
        dy, dx = 14 - cy, 14 - cx
        canvas = np.roll(canvas, dy, axis=0)
        canvas = np.roll(canvas, dx, axis=1)

    if debug:
        plt.imshow(canvas, cmap="gray")
        plt.title("Preprocessed image")
        plt.show()

    return canvas.reshape(-1).astype(np.float32)


def show_image(flat784: np.ndarray, title="image"):
    """Display a flattened 784 array as a 28x28 image."""
    plt.figure(figsize=(2.4, 2.4))
    plt.imshow(flat784.reshape(28, 28), cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


# ==========================
# Prediction Helper Methods
# ==========================

def predict_single(clf: Mnist, img_path: str, true_digit: int, show=True) -> bool:
    """Predict a single image and print Success/Fail line."""
    x = load_digit_image(img_path, force_resize=True)
    if show:
        show_image(x, title=f"{os.path.basename(img_path)} (expect {true_digit})")
    probs = clf.predict(x)
    pred = int(np.argmax(probs))
    if pred == int(true_digit):
        print(f"Success: Image {os.path.basename(img_path)} is for digit {true_digit} "
              f"is recognized as {pred}.", flush=True)
        return True
    else:
        print(f"Fail: Image {os.path.basename(img_path)} is for digit {true_digit} "
              f"but the inference result is {pred}.", flush=True)
        return False


def predict_all_in_folder(clf: Mnist, folder="test_images") -> None:
    """Evaluate all images n_m.png in test_images/ folder."""
    total, correct = 0, 0
    for n in range(10):
        for m in range(5):
            fname = f"{n}_{m}.png"
            path = os.path.join(folder, fname)
            if not os.path.exists(path):
                continue
            ok = predict_single(clf, path, true_digit=n, show=False)
            total += 1
            correct += int(ok)
    if total:
        print(f"\nHandwritten set accuracy: {correct}/{total} = {correct/total:.4f}")
    else:
        print("No test images found.")


# ===========
# Entry Point
# ===========

def main():
    parser = argparse.ArgumentParser(description="Test handwritten digits using pretrained MNIST model")
    parser.add_argument("image", nargs="?", help="Filename inside test_images/, e.g., 3_2.png")
    parser.add_argument("digit", nargs="?", type=int, help="True digit label (0-9)")
    parser.add_argument("--model", default="model/sample_weight.pkl", help="Path to pretrained model")
    parser.add_argument("--all", action="store_true", help="Test all images in test_images/")
    args = parser.parse_args()

    clf = Mnist(model_path=args.model)

    if args.all:
        # headless backend to suppress GUI popups
        matplotlib.use("Agg")
        predict_all_in_folder(clf, folder="test_images")
        return

    if not args.image or args.digit is None:
        print("Usage examples:\n"
              "  python module5-3.py 3_2.png 3\n"
              "  python module5-3.py --all")
        return

    img_path = os.path.join("test_images", args.image)
    predict_single(clf, img_path, args.digit, show=True)


if __name__ == "__main__":
    main()

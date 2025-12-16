import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from mnist_data import MnistData, softmax, stable_softmax

def show_examples(x, y, indices, title="Examples"):
    rows = 1
    cols = len(indices)
    plt.figure(figsize=(cols * 2.2, 2.4))
    for i, idx in enumerate(indices):
        img = x[idx].reshape(28, 28)
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap="gray")
        plt.title(f"idx {idx}\nlabel {int(y[idx])}")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    # --- Save plots to a folder when called via !python ---
    os.makedirs("figs", exist_ok=True)
    safe_name = title.replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join("figs", f"{safe_name}.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig=True)
    print(f"Saved figure: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root (where dataset dir lives)")
    parser.add_argument("--umid3", default=os.environ.get("UMID3", "351"),
                        help="First three digits of UMID (e.g., 351). Used to pick 3 images.")
    args = parser.parse_args()

    # Init MnistData (downloads on first run)
    mnist = MnistData(root=args.root, normalize=True, one_hot=False)
    (x_train, y_train), (x_test, y_test) = mnist.get_dataset()

    # Print shapes
    print("Train images:", x_train.shape, x_train.dtype)
    print("Train labels:", y_train.shape, y_train.dtype)
    print("Test images:", x_test.shape, x_test.dtype)
    print("Test labels:", y_test.shape, y_test.dtype)

    # Softmax demo
    a = np.array([1.0, 2.0, 3.0])
    print("softmax([1,2,3]) =", softmax(a))
    print("stable_softmax([1,2,3]) =", stable_softmax(a))


    # Show one train + one test image
    show_examples(x_train, y_train, [0], title="One Train Example")
    show_examples(x_test, y_test, [0], title="One Test Example")

    # Pick 3 train images based on UMID3 digits
    digits = [int(c) for c in str(args.umid3)[:3]]
    # Use digits as offsets to ensure we are in‑range
    idxs = [(d % len(x_train)) for d in digits]
    show_examples(x_train, y_train, idxs, title=f"Three Train Images (UMID3={args.umid3})")

if __name__ == "__main__":
    main()

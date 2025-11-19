import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from mnist_keras import MnistKeras, MnistKerasConfig


def load_and_preprocess_image(img_path):
    # 1. Load grayscale
    img = Image.open(img_path).convert("L")

    # 2. Resize to 28x28
    img = img.resize((28, 28), Image.LANCZOS)

    # 3. To numpy
    arr = np.array(img).astype("float32")

    # 4. Invert (if original is black digit on white background)
    arr = 255.0 - arr

    # 5. Normalize to [0, 1]
    arr /= 255.0

    # 6. Shape (1, 28, 28) for your Flatten input
    arr = np.expand_dims(arr, axis=0)

    return arr




def main():
    if len(sys.argv) != 3:
        print("Usage: python module8.py <image_file> <label>")
        sys.exit(1)

    img_path = sys.argv[1]
    true_label = int(sys.argv[2])

    # Load trained model ONCE
    config = MnistKerasConfig()
    mnist = MnistKeras(config)
    _, model = mnist.build_model(enforce_new=False)

    x = load_and_preprocess_image(img_path)

    preds = model.predict(x, verbose=0)
    pred_label = int(np.argmax(preds, axis=1)[0])

    # Show image
    plt.imshow(x[0], cmap="gray")
    plt.title(f"Predicted: {pred_label}, True: {true_label}")
    plt.axis("off")
    plt.show()

    # Success / Fail message
    if pred_label == true_label:
        print(
            f"Success: Image {img_path} is for digit {true_label}, "
            f"and the inference result is {pred_label}."
        )
    else:
        print(
            f"Fail: Image {img_path} is for digit {true_label}, "
            f"but the inference result is {pred_label}."
        )


if __name__ == "__main__":
    main()
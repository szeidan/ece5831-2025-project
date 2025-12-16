# TwoLayerNet with Backpropagation (MNIST Classification)

This project implements a **two-layer neural network (MLP)** trained from scratch using **NumPy**.  
The model learns to classify handwritten digits (0–9) from the **MNIST dataset** and can also test your own handwritten digits.

---

## 📂 Project Structure

```
twolayernet_mnist/
│
├── mnist_data.py                # Loads and preprocesses MNIST data (flatten + one-hot)
├── layer.py                     # Core neural network layers (Affine, ReLU, SoftmaxWithLoss)
├── two_layer_net_with_backprop.py # Two-layer MLP with forward/backward implementation
├── utility.py                   # Helper functions (one-hot, accuracy, save/load pickle)
├── train.py                     # Trains the network and saves the model
├── module7.py                   # Loads model and tests on MNIST or custom images
├── module7.ipynb                # Notebook version for training + testing
└── test_images/                 # Folder for your handwritten digits (optional)
```

---

## 🚀 How to Run

### 1. Install Requirements
Make sure Python ≥ 3.8 is installed, then install dependencies:
```bash
pip install numpy pillow tensorflow
```
*(TensorFlow is only used for loading MNIST; scikit-learn is an optional fallback.)*

---

### 2. Train the Model
Run the training script — this will download MNIST, train for a few epochs,  
and save the trained weights to **`zeidan_mnist_model.pkl`**.

```bash
python train.py
```

You can adjust training parameters (epochs, learning rate, hidden size) inside `train.py`.

---

### 3. Test on MNIST (and Your Own Digits)
After training, test your model with:
```bash
python module7.py
```
This will:
- Print test accuracy on the MNIST test set  
- Evaluate all images in the `test_images/` folder (e.g. `0_0.png`, `1_2.png`, etc.)
- Save preprocessed previews in `preprocessed_preview/` for inspection

---

### 4. Notebook Option
You can also open **`module7.ipynb`** in Jupyter or VS Code to train and test interactively.

---

## 🖼️ Custom Digit Testing
Place your own digit images in the `test_images/` folder using names like:
```
0_0.png, 0_1.png, ..., 9_4.png
```

Each image:
- Should be roughly square (white background, dark digit)
- Will be automatically resized and centered to 28×28 pixels
- Is normalized to match MNIST format

---

## 🧠 Notes
- Model is purely NumPy-based (no PyTorch or TensorFlow training)
- You can tweak:
  - `hidden_size` → number of neurons in hidden layer
  - `epochs` → training duration
  - `reg` → L2 regularization strength


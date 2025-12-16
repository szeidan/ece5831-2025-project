# MNIST Data and Handwritten Digit Recognition

This project implements a full MNIST workflow for ECE 5831 — Pattern Recognition and Neural Networks.

## Files
- **module5-2.ipynb** – Jupyter Notebook for building and testing the MnistData class.
- **module5-2.py** – Script version to test MnistData and visualize example images.
- **mnist_data.py** – MnistData class for downloading, parsing, normalizing, and one-hot encoding the MNIST dataset.
- **mnist.py** – Mnist class implementing a 3-layer MLP using pretrained weights (sample_weight.pkl).
- **module5-3.ipynb** – Notebook for testing the Mnist class and handwritten digit recognition.
- **module5-3.py** – Script to test handwritten digit images and print Success/Fail results.
- **model/sample_weight.pkl** – Pretrained MLP model weights.
- **test_images/** – Contains 50 user-created handwritten digit images (five per digit 0–9).

## How to Run
python module5-3.py --all
python module5-3.py 3_2.png 3
### 1. Load and Test MNIST Data
python module5-2.py --umid3 351

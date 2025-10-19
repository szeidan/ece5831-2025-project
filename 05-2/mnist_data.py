
import os
import gzip
import pickle
import urllib.request
from typing import Tuple, Dict, Any

import numpy as np


class MnistData:
    """
    Downloads the MNIST dataset (gz files), converts them to NumPy arrays,
    normalizes images to [0, 1], supports one‑hot encoding, and caches to mnist.pkl.
    """
    url_base = "http://jrkwon.com/data/ece5831/mnist/"
    key_to_file = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz",
    }
    dataset_dir = "dataset"
    mnist_pickle_name = "mnist.pkl"
    img_size = 28 * 28
    num_classes = 10

    def __init__(self, root: str = ".", normalize: bool = True, one_hot: bool = False):
        self.root = os.path.abspath(root)
        self.normalize = normalize
        self.one_hot = one_hot

        self._ensure_dirs()
        self._download_all()
        self.dataset = self._make_dataset()

    # ------------------ Public API ------------------
    def get_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Returns: (train_images, train_labels), (test_images, test_labels)
        Images are float32 (if normalize=True) else uint8.
        Labels are int64 by default or one‑hot (float32) if one_hot=True.
        """
        ds = self.dataset
        x_train = ds["train_images"]
        y_train = ds["train_labels"]
        x_test  = ds["test_images"]
        y_test  = ds["test_labels"]
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def to_one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
        """Convert integer labels to one‑hot rows."""
        labels = labels.astype(np.int64).ravel()
        out = np.zeros((labels.size, num_classes), dtype=np.float32)
        out[np.arange(labels.size), labels] = 1.0
        return out

    # ------------------ Internal helpers ------------------
    def _ensure_dirs(self) -> None:
        os.makedirs(os.path.join(self.root, self.dataset_dir), exist_ok=True)

    def _pickle_path(self) -> str:
        return os.path.join(self.root, self.dataset_dir, self.mnist_pickle_name)

    def _gz_path(self, filename: str) -> str:
        return os.path.join(self.root, self.dataset_dir, filename)

    def _download(self, filename: str) -> None:
        """Download a single gz file if missing (adds empty Accept header to avoid HTTP 406)."""
        dst = self._gz_path(filename)
        if os.path.exists(dst):
            return
        url = self.url_base + filename
        opener = urllib.request.build_opener()
        opener.addheaders = [("Accept", "")]
        urllib.request.install_opener(opener)
        print(f"Downloading {url} -> {dst}")
        urllib.request.urlretrieve(url, dst)
        print("Done.")

    def _download_all(self) -> None:
        """Download all four gz files using key_to_file mapping."""
        for fname in self.key_to_file.values():
            self._download(fname)

    def _load_labels(self, file_path: str) -> np.ndarray:
        """Parse IDX label file into shape (N,) uint8 array."""
        with gzip.open(file_path, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    def _load_images(self, file_path: str) -> np.ndarray:
        """Parse IDX image file into shape (N, 784) uint8 array."""
        with gzip.open(file_path, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        imgs = data.reshape(-1, self.img_size)
        return imgs

    def _convert_numpy(self) -> Dict[str, np.ndarray]:
        """Load all four gz files -> dict with train/test images/labels (uint8)."""
        dset: Dict[str, np.ndarray] = {}
        dset["train_images"] = self._load_images(self._gz_path(self.key_to_file["train_images"]))
        dset["train_labels"] = self._load_labels(self._gz_path(self.key_to_file["train_labels"]))
        dset["test_images"]  = self._load_images(self._gz_path(self.key_to_file["test_images"]))
        dset["test_labels"]  = self._load_labels(self._gz_path(self.key_to_file["test_labels"]))
        return dset

    def _make_dataset(self) -> Dict[str, Any]:
        """
        If mnist.pkl exists, load and return it. Otherwise, convert from gz and pickle it.
        Applies normalization and one‑hot encoding per flags.
        """
        pkl_path = self._pickle_path()
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                dset = pickle.load(f)
        else:
            dset = self._convert_numpy()
            with open(pkl_path, "wb") as f:
                pickle.dump(dset, f)

        # Normalize if requested
        if self.normalize:
            for k in ("train_images", "test_images"):
                dset[k] = dset[k].astype(np.float32) / 255.0

        # One‑hot if requested
        if self.one_hot:
            for k in ("train_labels", "test_labels"):
                dset[k] = self.to_one_hot(dset[k], self.num_classes)

        return dset

# --- Utility functions for softmax ---
def softmax(x, axis=-1):
    """
    Numerically stable softmax.
    - Works for 1D or ND arrays (default normalize over last axis).
    - Avoids overflow by subtracting the per-axis max before exp.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    shifted = x - m
    ex = np.exp(shifted)
    denom = np.sum(ex, axis=axis, keepdims=True)
    denom = np.clip(denom, a_min=np.finfo(ex.dtype).tiny, a_max=None)
    out = ex / denom
    # if original input was 1D and axis=-1, return 1D
    if out.ndim == 2 and axis == -1 and x.ndim == 1:
        return out[0]
    return out

# Backward-compatible alias for older code that imported stable_softmax
stable_softmax = softmax
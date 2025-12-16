import gzip
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import urllib.request

class MnistData:
    
    # dictionary to map a file type to its file name
    key_to_file = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }
    mnist_data_dir = "mnist"    # location for .gz files
    mnist_pickle_name = "mnist.pkl"
    base_url = 'http://jrkwon.com/data/ece5831/mnist/'

    def __init__(self):
        self._download_all()
        self.dataset = self._make_dataset()

    # Function to download MNIST data
    def _download(self, filename):

        if os.path.exists(self.mnist_data_dir) is False:
            os.makedirs(self.mnist_data_dir, exist_ok=True)

        print(f"Downloading {filename}...")
        filepath = os.path.join(self.mnist_data_dir, filename)

        # check if file already exists
        if os.path.exists(filepath):
            print(f"{filename} already exists. Skipping download.")
            return

        # to resolve 406 Not Acceptable error
        opener = urllib.request.build_opener()
        opener.addheaders = [('Accept', '')]
        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(self.base_url + filename, filepath)

        print("Download complete.")

    def _download_all(self):

        for key in self.key_to_file:
            self._download(self.key_to_file[key])

    # Function to load MNIST images
    def _load_images(self, file_path):

        with gzip.open(file_path, 'rb') as f:
            f.read(16)  # Skip the header
            buf = f.read()  # Read the rest of the file
            images = np.frombuffer(buf, dtype=np.uint8)
            images = images.reshape(-1, 28, 28)  # Reshape to (num_images, 28, 28)
        return images

    # Function to load MNIST labels
    def _load_labels(self, file_path):

        with gzip.open(file_path, 'rb') as f:
            f.read(8)  # Skip the header
            buf = f.read()  # Read the rest of the file
            labels = np.frombuffer(buf, dtype=np.uint8)
        return labels
    
    def _make_dataset(self):
        if os.path.exists(self.mnist_pickle_name):
            print(f"{self.mnist_pickle_name} already exists. Loading dataset from pickle file.")
            with open(self.mnist_pickle_name, 'rb') as f:
                mnist_data = pickle.load(f)
            return mnist_data

        # Load training and test data
        train_images = self._load_images(os.path.join(self.mnist_data_dir, self.key_to_file["train_images"]))
        train_labels = self._load_labels(os.path.join(self.mnist_data_dir, self.key_to_file["train_labels"]))
        test_images = self._load_images(os.path.join(self.mnist_data_dir, self.key_to_file["test_images"]))
        test_labels = self._load_labels(os.path.join(self.mnist_data_dir, self.key_to_file["test_labels"]))

        # Package the data into a dictionary
        mnist_data = {
            "train_images": train_images,
            "train_labels": train_labels,
            "test_images": test_images,
            "test_labels": test_labels
        }

        # Save the dataset as a pickle file
        with open(self.mnist_pickle_name, 'wb') as f:
            pickle.dump(mnist_data, f)

        print(f"MNIST dataset created and saved to {self.mnist_pickle_name}")
        return mnist_data
    
    def get_dataset(self):
        # normalize images to [0, 1]
        self.dataset['train_images'] = self.dataset['train_images'].astype(np.float32)
        self.dataset['train_images'] /= 255.0
        self.dataset['test_images'] = self.dataset['test_images'].astype(np.float32)
        self.dataset['test_images'] /= 255.0
        
        return (self.dataset['train_images'], self.dataset['train_labels']), \
                (self.dataset['test_images'], self.dataset['test_labels'])
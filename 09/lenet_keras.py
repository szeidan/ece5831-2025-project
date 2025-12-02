
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class LeNetKerasConfig:
    def __init__(self):
        self.batch_size = 32
        self.num_classes = 10
        self.epochs = 10
        self.img_rows, self.img_cols = 28, 28
        self.input_shape = (self.img_rows, self.img_cols, 1)
        self.loss_function = "sparse_categorical_crossentropy"
        self.optimizer = "adam"
        self.metrics = ["accuracy"]
        self.validation_split = 0.2
        self.verbose = 1
        self.model_file_path = "lenet_keras.h5"


class LeNetKeras:
    def __init__(self, config: LeNetKerasConfig):
        self.config = config

    def load_data(self):
        """Load and normalize MNIST, add channel dimension for Conv2D."""
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        # Add channel dimension: (N, 28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        return (x_train, y_train), (x_test, y_test)

    # returns a tuple (new, model): new is True if a new model is created.
    def build_model(self, enforce_new: bool = False):
        if os.path.exists(self.config.model_file_path) and not enforce_new:
            model = keras.models.load_model(self.config.model_file_path)
            if self.config.verbose:
                print(f"Model - {self.config.model_file_path} - loaded from file.")
            return False, model

        # Classic LeNet-5 style architecture
        model = keras.Sequential(
            [
                layers.Conv2D(
                    6,
                    kernel_size=(5, 5),
                    activation="relu",
                    input_shape=self.config.input_shape,
                ),
                layers.AveragePooling2D(),
                layers.Conv2D(16, kernel_size=(5, 5), activation="relu"),
                layers.AveragePooling2D(),
                layers.Flatten(),
                layers.Dense(120, activation="relu"),
                layers.Dense(84, activation="relu"),
                layers.Dense(self.config.num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=self.config.optimizer,
            loss=self.config.loss_function,
            metrics=self.config.metrics,
        )
        return True, model

    def train(self, model, x_train, y_train):
        history = model.fit(
            x_train,
            y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=self.config.validation_split,
            verbose=self.config.verbose,
        )
        model.save(self.config.model_file_path)
        if self.config.verbose:
            print(f"Model saved to {self.config.model_file_path}")
        return history

    def evaluate(self, model, x_test, y_test):
        return model.evaluate(x_test, y_test, verbose=self.config.verbose)

    def predict(self, model, x):
        """x is expected to be shape (N, 28, 28, 1) float32 in [0,1]."""
        return model.predict(x, verbose=0)

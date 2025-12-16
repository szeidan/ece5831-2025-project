import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras


class MnistKerasConfig:
    def __init__(self):
        self.batch_size = 32
        self.num_classes = 10
        self.epochs = 12
        self.img_rows, self.img_cols = 28, 28
        self.input_shape = (self.img_rows, self.img_cols, 1)
        self.loss_function = 'sparse_categorical_crossentropy'
        self.optimizer = 'adam'
        self.metrics = ['accuracy']
        self.layers = [100, 50]
        self.validation_split = 0.2
        self.verbose = 1
        self.model_file_path = "mnist_keras_model.h5"

class MnistKeras:
    def __init__(self, config: MnistKerasConfig):
        self.config = config

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        return (x_train, y_train), (x_test, y_test)
    
    # returns a tuple (new, model): new is True if a new model is created.
    def build_model(self, enforce_new=False):
        if os.path.exists(self.config.model_file_path) and not enforce_new:
            model = keras.models.load_model(self.config.model_file_path)
            if self.config.verbose:
                print(f'Model - {self.config.model_file_path} - loaded from file.')
            return False, model
        
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(self.config.img_rows, self.config.img_cols)),
            keras.layers.Dense(self.config.layers[0], activation='relu'),
            keras.layers.Dense(self.config.layers[1], activation='relu'),
            keras.layers.Dense(self.config.num_classes, activation='softmax')])

        model.compile(optimizer=self.config.optimizer,
                    loss=self.config.loss_function,
                    metrics=self.config.metrics)
        return True, model
    
    def train(self, model, x_train, y_train):
        history = model.fit(x_train, y_train,
                            batch_size=self.config.batch_size,
                            epochs=self.config.epochs,
                            validation_split=self.config.validation_split,
                            verbose=self.config.verbose)
        model.save(self.config.model_file_path)
        if self.config.verbose:
            print(f"Model saved to {self.config.model_file_path}")

        return history
    
    def plot_history(self, history):
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()

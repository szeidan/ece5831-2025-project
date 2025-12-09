
from tensorflow import keras
from dogs_cats import DogsCats

class DogsCatsAug(DogsCats):
    def build_network(self):
        input_shape = self.config.image_shape
        num_classes = 2

        aug = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.2),
            keras.layers.RandomZoom(0.2)
        ])

        inputs = keras.Input(shape=input_shape)
        x = aug(inputs)
        x = keras.layers.Rescaling(1./255)(x)
        x = keras.layers.Conv2D(32, 3, activation="relu")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(64, 3, activation="relu")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(128, 3, activation="relu")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model = model
        return model

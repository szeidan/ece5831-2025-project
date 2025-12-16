
from tensorflow import keras
from dogs_cats import DogsCats

class DogsCatsPre(DogsCats):
    def build_network(self):
        input_shape = self.config.image_shape
        num_classes = 2

        base_model = keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
        base_model.trainable = False

        inputs = keras.Input(shape=input_shape)
        x = keras.applications.vgg16.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(1e-4),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        self.model = model
        return model

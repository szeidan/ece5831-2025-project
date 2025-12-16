from tensorflow import keras
from dogs_cats_pre import DogsCatsPre

class DogsCatsFine(DogsCatsPre):
    def fine_tune(self, model_name: str):

        # 1) Find the VGG16 base model inside the full model
        base_model = None
        for layer in self.model.layers:
            # In this architecture VGG16 should appear as a sub-model named "vgg16"
            if isinstance(layer, keras.Model) or "vgg16" in layer.name.lower():
                base_model = layer
                break

        if base_model is None:
            raise RuntimeError("Could not find VGG16 base model inside DogsCatsFine.model")

        # 2) Unfreeze only the block5 convolution layers
        for layer in base_model.layers:
            if layer.name.startswith("block5_conv"):
                layer.trainable = True
            else:
                layer.trainable = False

        # 3) Recompile with a smaller learning rate for fine-tuning
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # 4) Fine-tune using the existing datasets
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=model_name,
                save_best_only=True,
                monitor="val_accuracy",
                mode="max",
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
            ),
        ]

        history = self.model.fit(
            self.train_dataset,
            validation_data=self.valid_dataset,
            epochs=self.config.epochs,
            callbacks=callbacks,
        )

        # 5) Evaluate on test set
        test_loss, test_acc = self.model.evaluate(self.test_dataset)
        print(f"[Fine-tune] Test accuracy: {test_acc:.3f}")

        return history

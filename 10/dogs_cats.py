
from tensorflow import keras
import pathlib, os, shutil

class DogsCatsConfig:
    def __init__(self):
        self.base_dir = pathlib.Path("data/dogs-vs-cats")
        self.kaggle_base_dir = pathlib.Path("data/kaggle/train")
        self.num_train = 6000
        self.num_valid = 2000
        self.num_test = 2000
        self.categories = ["cat", "dog"]

        self.train_dir = self.base_dir / "train"
        self.valid_dir = self.base_dir / "valid"
        self.test_dir = self.base_dir / "test"

        self.image_shape = (180, 180, 3)
        self.batch_size = 32
        self.epochs = 20

class DogsCats:
    def __init__(self, config: DogsCatsConfig):
        self.config = config
        self._prepare_datasets()
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.model = None

    def _make_datasets_directories(self):
        for category in self.config.categories:
            os.makedirs(self.config.train_dir / category, exist_ok=True)
            os.makedirs(self.config.valid_dir / category, exist_ok=True)
            os.makedirs(self.config.test_dir / category, exist_ok=True)

    def _copy_datasets(self):
        for category in self.config.categories:
            for i in range(self.config.num_train):
                src = self.config.kaggle_base_dir / f"{category}.{i}.jpg"
                dst = self.config.train_dir / category / f"{category}.{i}.jpg"
                if not dst.exists(): shutil.copyfile(src, dst)

            for i in range(self.config.num_valid):
                src = self.config.kaggle_base_dir / f"{category}.{6000+i}.jpg"
                dst = self.config.valid_dir / category / f"{category}.{6000+i}.jpg"
                if not dst.exists(): shutil.copyfile(src, dst)

            for i in range(self.config.num_test):
                src = self.config.kaggle_base_dir / f"{category}.{8000+i}.jpg"
                dst = self.config.test_dir / category / f"{category}.{8000+i}.jpg"
                if not dst.exists(): shutil.copyfile(src, dst)

    def _prepare_datasets(self):
        if not self.config.train_dir.exists():
            self._make_datasets_directories()
            self._copy_datasets()

    def make_datasets(self):
        cfg = self.config
        self.train_dataset = keras.utils.image_dataset_from_directory(
            cfg.train_dir, image_size=cfg.image_shape[:2], batch_size=cfg.batch_size
        )
        self.valid_dataset = keras.utils.image_dataset_from_directory(
            cfg.valid_dir, image_size=cfg.image_shape[:2], batch_size=cfg.batch_size
        )
        self.test_dataset = keras.utils.image_dataset_from_directory(
            cfg.test_dir, image_size=cfg.image_shape[:2], batch_size=cfg.batch_size
        )

    def build_network(self):
        input_shape = self.config.image_shape
        num_classes = 2

        model = keras.Sequential([
            keras.layers.Rescaling(1./255, input_shape=input_shape),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax")
        ])

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model = model
        return model

    def train_model(self, model_name):
        history = self.model.fit(self.train_dataset, validation_data=self.valid_dataset, epochs=self.config.epochs)
        self.model.save(model_name)
        test_loss, test_acc = self.model.evaluate(self.test_dataset)
        print("Test accuracy:", test_acc)
        return history

    def load_model(self, model_name):
        self.model = keras.models.load_model(model_name)

    def predict(self, image_file):
        img = keras.utils.load_img(image_file, target_size=self.config.image_shape[:2])
        x = keras.utils.img_to_array(img)
        x = x[None, ...]
        preds = self.model.predict(x)
        return self.config.categories[preds.argmax()]

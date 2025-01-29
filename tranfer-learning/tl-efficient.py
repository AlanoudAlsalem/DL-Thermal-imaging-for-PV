import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class ThermalImageClassifierEfficientNet:
    def __init__(self, train_dir: str, val_dir: str, test_dir: str, num_classes: int = 12):
        self.num_classes = num_classes
        self.height = 75
        self.width = int(self.height * (320/240))
        self.image_size = (self.height, self.width)

        # Check GPU availability
        self._setup_gpu()

        # Initialize datasets with smaller batch size
        self.train_ds = self._prepare_dataset(train_dir, batch_size=16)  # Reduced batch size
        self.val_ds = self._prepare_dataset(val_dir, batch_size=16)
        self.test_ds = self._prepare_dataset(test_dir, batch_size=16)

        self.model = None
        self.history = None

    def _setup_gpu(self):
        """Configure GPU if available."""
        if not tf.config.list_physical_devices('GPU'):
            print("No GPU detected. Training will proceed on CPU.")
        else:
            print(f"Using GPU: {tf.config.list_physical_devices('GPU')}")

    @tf.autograph.experimental.do_not_convert
    def _grey_to_rgb(self, image):
        """Convert grayscale images to RGB if needed."""
        return tf.image.grayscale_to_rgb(image) if image.shape[-1] == 1 else image

    def _prepare_dataset(self, directory: str, batch_size: int = 16):
        """Prepare and augment the dataset with normalization."""
        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            image_size=self.image_size,
            batch_size=batch_size,
            shuffle=True  # Added shuffle
        )

        # Normalize images and apply one-hot encoding
        ds = ds.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.one_hot(y, self.num_classes))
        ).map(
            lambda x, y: (self._grey_to_rgb(x), y)
        )

        return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def build_model(self):
        """Build and compile the model architecture using EfficientNetV2B3."""
        # Data augmentation layers
        input_layer = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        augmentation = self._create_augmentation_model(input_layer)

        # Load pre-trained EfficientNetV2B3
        pretrained = tf.keras.applications.EfficientNetV2B3(
            include_top=False,
            weights='imagenet',
            input_shape=(self.height, self.width, 3)
        )
        pretrained.trainable = False

        # Build final model with dropout for regularization
        x = augmentation.output
        x = pretrained(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)  # Added dropout
        x = tf.keras.layers.Dense(256, activation='relu')(x)  # Added intermediate layer
        x = tf.keras.layers.Dropout(0.2)(x)  # Added dropout
        output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)  # Changed to softmax

        self.model = tf.keras.models.Model(inputs=augmentation.input, outputs=output)

        self._compile_model()

    def _create_augmentation_model(self, input_layer):
        """Create data augmentation pipeline with additional augmentations."""
        x = tf.keras.layers.Resizing(self.height, self.width)(input_layer)
        x = tf.keras.layers.RandomBrightness((-0.2, 0.2))(x)  # Reduced range
        x = tf.keras.layers.RandomContrast(0.2)(x)  # Reduced contrast
        x = tf.keras.layers.RandomFlip()(x)
        x = tf.keras.layers.RandomRotation(0.1)(x)  # Added rotation
        x = tf.keras.layers.RandomZoom(0.1)(x)  # Added zoom
        x = tf.keras.layers.Lambda(
            tf.keras.applications.efficientnet_v2.preprocess_input
        )(x)

        return tf.keras.models.Model(inputs=input_layer, outputs=x)

    def _compile_model(self, learning_rate: float = 1e-3):
        """Compile the model with updated metrics."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Changed to Adam
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
            ]
        )

    def train(self, epochs: int = 40, model_path: str = "thermal_model_efficientnet.keras"):
        """Train the model with updated callbacks."""
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                save_best_only=True,
                monitor="val_accuracy"  # Updated monitor metric
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",  # Changed to val_loss
                factor=0.5,  # Less aggressive reduction
                patience=5,
                verbose=1,
                min_lr=1e-6
            ),
            tf.keras.callbacks.EarlyStopping(  # Added early stopping
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks
        )

    def fine_tune(self, start_layer: int = -50):  # Changed to use negative indexing
        """Fine-tune the last layers of the model."""
        for layer in self.model.layers[start_layer:]:
            layer.trainable = True

        self._compile_model(learning_rate=1e-5)  # Even lower learning rate for fine-tuning

    # Rest of the methods remain the same
    def evaluate(self, dataset='val'):
        """
        Evaluate the model on the specified dataset.

        Args:
            dataset: Which dataset to evaluate on ('train', 'val', or 'test')
        """
        eval_ds = {
            'train': self.train_ds,
            'val': self.val_ds,
            'test': self.test_ds
        }.get(dataset)

        return self.model.evaluate(eval_ds)

    def save_model(self, path: str):
        """Save the trained model."""
        self.model.save(path)

    @classmethod
    def load_model(cls, path: str):
        """Load a saved model."""
        return tf.keras.models.load_model(
            path,
            custom_objects={
                'preprocess_input': tf.keras.applications.efficientnet_v2.preprocess_input
            }
        )

    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available. Train the model first.")
            return

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

# Usage example:
if __name__ == "__main__":
    # Initialize paths
    base_dir = Path("Dataset/")
    model_path = base_dir / "EfficientNetV2B3.keras"

    # Create classifier instance
    classifier = ThermalImageClassifierEfficientNet(
        train_dir=str(base_dir / "train"),
        val_dir=str(base_dir / "val"),
        test_dir=str(base_dir / "test")
    )

    # Train model
    classifier.build_model()
    classifier.train(epochs=1, model_path=str(model_path))

    # Plot training progress
    classifier.plot_training_history()

    # Fine-tune model
    classifier.fine_tune(start_layer=402)
    classifier.train(epochs=40, model_path=str(model_path))

    # Evaluate and save
    classifier.evaluate('val')
    classifier.save_model(str(model_path))

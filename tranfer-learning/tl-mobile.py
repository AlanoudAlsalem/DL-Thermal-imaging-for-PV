import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class ThermalImageClassifierMobileNet:
    def __init__(self, train_dir: str, val_dir: str, test_dir: str, num_classes: int = 12):
        self.num_classes = num_classes
        self.height = 75
        self.width = int(self.height * (320/240))
        self.image_size = (self.height, self.width)
        
        self._setup_gpu()
        
        # Initialize datasets with smaller batch size and proper normalization
        self.train_ds = self._prepare_dataset(train_dir, batch_size=16)
        self.val_ds = self._prepare_dataset(val_dir, batch_size=16)
        self.test_ds = self._prepare_dataset(test_dir, batch_size=16)
        
        self.model = None
        self.history = None

    def _setup_gpu(self):
        if not tf.config.list_physical_devices('GPU'):
            print("No GPU detected. Training will proceed on CPU.")
        else:
            print(f"Using GPU: {tf.config.list_physical_devices('GPU')}")

    @tf.autograph.experimental.do_not_convert
    def _grey_to_rgb(self, image):
        return tf.image.grayscale_to_rgb(image) if image.shape[-1] == 1 else image

    def _prepare_dataset(self, directory: str, batch_size: int = 16):
        """Prepare dataset with proper normalization and augmentation."""
        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            image_size=self.image_size,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Normalize images and apply one-hot encoding
        ds = ds.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.one_hot(y, self.num_classes))
        ).map(
            lambda x, y: (self._grey_to_rgb(x), y)
        )
        
        return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def build_model(self):
        """Build model with improved architecture."""
        # Create input layer
        input_layer = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        
        # Create augmentation layers
        augmented = tf.keras.layers.Resizing(self.height, self.width)(input_layer)
        augmented = tf.keras.layers.RandomBrightness((-0.2, 0.2))(augmented)
        augmented = tf.keras.layers.RandomContrast(0.2)(augmented)
        augmented = tf.keras.layers.RandomFlip(mode='horizontal')(augmented)
        augmented = tf.keras.layers.RandomRotation(0.1)(augmented)
        augmented = tf.keras.layers.RandomZoom(0.1)(augmented)
        augmented = tf.keras.layers.Lambda(
            tf.keras.applications.mobilenet_v2.preprocess_input
        )(augmented)
        
        # Load pre-trained MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(self.height, self.width, 3)
        )
        base_model.trainable = False
        
        # Build the complete model
        x = base_model(augmented)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create and compile the model
        self.model = tf.keras.Model(inputs=input_layer, outputs=outputs)
        self._compile_model()
        
        return self.model

    def _compile_model(self, learning_rate: float = 1e-3):
        """Compile model with proper metrics."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
            ]
        )

    def train(self, epochs: int = 40, model_path: str = "thermal_model_mobilenet.keras"):
        """Train with improved callbacks."""
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")
            
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                save_best_only=True,
                monitor="val_accuracy"
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-6
            ),
            tf.keras.callbacks.EarlyStopping(
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
        
        return self.history

    def fine_tune(self, num_layers: int = 50):
        """Improved fine-tuning approach."""
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")
            
        # Find the MobileNetV2 base model
        for layer in self.model.layers:
            if isinstance(layer, type(tf.keras.applications.MobileNetV2())):
                base_model = layer
                break
        else:
            raise ValueError("Could not find MobileNetV2 base model in layers.")
        
        # Unfreeze the last num_layers layers
        for layer in base_model.layers[-num_layers:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
            
        self._compile_model(learning_rate=1e-5)

    def evaluate(self, dataset='test'):
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")
            
        eval_ds = {
            'train': self.train_ds,
            'val': self.val_ds,
            'test': self.test_ds
        }.get(dataset)
        
        if eval_ds is None:
            raise ValueError("Dataset must be one of: 'train', 'val', 'test'")
        
        results = self.model.evaluate(eval_ds)
        metrics = {name: value for name, value in zip(self.model.metrics_names, results)}
        return metrics
        
    def save_model(self, path: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Build and train the model first.")
        self.model.save(path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str):
        """Load a saved model."""
        return tf.keras.models.load_model(
            path,
            custom_objects={
                'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input
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
    model_path = base_dir / "Mobile.h5"

    # Create classifier instance
    classifier = ThermalImageClassifierMobileNet(
        train_dir=str(base_dir / "train"),
        val_dir=str(base_dir / "val"),
        test_dir=str(base_dir / "test")
    )

    # Build and train model
    classifier.build_model()
    classifier.train(epochs=40, model_path=str(model_path))

    # Plot training progress
    classifier.plot_training_history()

    # Optional: Fine-tune and continue training
    classifier.fine_tune(num_layers=50)
    classifier.train(epochs=20, model_path=str(model_path))

    # Evaluate and save
    metrics = classifier.evaluate('test')
    print("Test metrics:", metrics)
    classifier.save_model(str(model_path))

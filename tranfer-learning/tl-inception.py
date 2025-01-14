import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class ThermalImageClassifier:
    def __init__(self, train_dir: str, val_dir: str, test_dir: str, num_classes: int = 12):
        self.num_classes = num_classes
        # InceptionV3 requires input size to be at least 75x75
        self.height = 149  # Changed to match InceptionV3 requirements better
        self.width = int(self.height * (320/240))
        self.image_size = (self.height, self.width)
        
        self._setup_gpu()
        
        # Initialize datasets with smaller batch size
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
            shuffle=True  # Added shuffle
        )
        
        # Normalize images and apply one-hot encoding
        ds = ds.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.one_hot(y, self.num_classes))
        ).map(
            lambda x, y: (self._grey_to_rgb(x), y)
        )
        
        return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def _create_augmentation_model(self, input_layer):
    """Create data augmentation pipeline with sequential layers."""
        augmentation = tf.keras.Sequential([
            tf.keras.layers.Resizing(self.height, self.width),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(factor=(-0.2, 0.2)),
            tf.keras.layers.RandomContrast(factor=(0.8, 1.2)),
            tf.keras.layers.Lambda(tf.keras.applications.inception_v3.preprocess_input)
            ])
    
        return tf.keras.models.Model(inputs=input_layer, outputs=augmentation(input_layer))

# Usage in the build_model method:
    def build_model(self):
    """Build and compile the model architecture."""
        input_layer = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        augmentation = self._create_augmentation_model(input_layer)
    
    # Load pre-trained InceptionV3
        pretrained = tf.keras.applications.InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=(self.height, self.width, 3)
            )
        pretrained.trainable = False
    
    # Build final model with improved architecture
        x = augmentation.output
        x = pretrained(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
    
        self.model = tf.keras.models.Model(inputs=augmentation.input, outputs=output)
        self._compile_model()

    def _compile_model(self, learning_rate: float = 1e-3):
        """Compile model with proper metrics."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Changed to Adam
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
            ]
        )

    def train(self, epochs: int = 40, model_path: str = "thermal_model_inception.keras"):
        """Train with improved callbacks."""
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                save_best_only=True,
                monitor="val_accuracy"  # Fixed monitor metric
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
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

    def fine_tune(self, num_layers: int = 20):
        """Improved fine-tuning strategy."""
        # Unfreeze the last num_layers layers
        for layer in self.model.layers[2].layers[-num_layers:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
            
        self._compile_model(learning_rate=1e-5)  # Lower learning rate for fine-tuning
        
    def evaluate(self, dataset='test'):
        """Evaluate the model with proper metrics."""
        eval_ds = {
            'train': self.train_ds,
            'val': self.val_ds,
            'test': self.test_ds
        }.get(dataset)
        
        return self.model.evaluate(eval_ds)
    
    def evaluate(self):
        """Evaluate the model on the test dataset."""
        return self.model.evaluate(self.test_ds)

    def save_model(self, path: str):
        """Save the trained model."""
        self.model.save(path)

    @classmethod
    def load_model(cls, path: str):
        """Load a saved model."""
        return tf.keras.models.load_model(
            path,
            custom_objects={
                'preprocess_input': tf.keras.applications.inception_v3.preprocess_input
            }
        )

# Usage example:
if __name__ == "__main__":
    # Initialize paths
    base_dir = Path("/content/drive/MyDrive/Thermal Imaging Project/Greyscale Images/Dataset")
    model_path = base_dir / "InceptionV3.keras"

    # Create classifier instance
    classifier = ThermalImageClassifier(
        train_dir=str(base_dir / "train"),
        val_dir=str(base_dir / "val"),
        test_dir=str(base_dir / "test")
    )

    # Train model
    classifier.build_model()
    classifier.train(epochs=40, model_path=str(model_path))

    # Fine-tune model
    classifier.fine_tune(num_layers=2)
    classifier.train(epochs=40, model_path=str(model_path))

    # Evaluate and save
    classifier.evaluate()
    classifier.save_model(str(model_path))

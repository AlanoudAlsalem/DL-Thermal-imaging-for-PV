import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class ThermalImageClassifierResNet:
    def __init__(self, train_dir: str, val_dir: str, test_dir: str, num_classes: int = 12):
        self.num_classes = num_classes
        # Increased size for ResNet50
        self.height = 224  # Changed to standard ResNet input size
        self.width = int(self.height * (320/240))
        self.image_size = (self.height, self.width)
        
        self._setup_gpu()
        
        # Initialize datasets with smaller batch size
        self.train_ds = self._prepare_dataset(train_dir, batch_size=16)
        self.val_ds = self._prepare_dataset(val_dir, batch_size=16)
        self.test_ds = self._prepare_dataset(test_dir, batch_size=16)
        
        self.model = None
        self.history = None
        self.class_weights = None

    def _setup_gpu(self):
        if not tf.config.list_physical_devices('GPU'):
            print("No GPU detected. Training will proceed on CPU.")
        else:
            print(f"Using GPU: {tf.config.list_physical_devices('GPU')}")

    @tf.autograph.experimental.do_not_convert
    def _grey_to_rgb(self, image):
        return tf.image.grayscale_to_rgb(image) if image.shape[-1] == 1 else image

    def _prepare_dataset(self, directory: str, batch_size: int = 16):
        """Prepare dataset with proper normalization."""
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

    def calculate_class_weights(self):
        """Calculate balanced class weights."""
        labels = []
        for _, y in self.train_ds:
            labels.append(tf.argmax(y, axis=1))
        labels = np.concatenate(labels)
        
        # Calculate class weights using sklearn's formula
        n_samples = len(labels)
        n_classes = self.num_classes
        
        # Count samples per class
        counts = np.bincount(labels, minlength=n_classes)
        weights = n_samples / (n_classes * counts)
        
        self.class_weights = dict(enumerate(weights))
        return self.class_weights

    def build_model(self):
        """Build model with improved architecture."""
        input_layer = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        augmentation = self._create_augmentation_model(input_layer)
        
        # Load pre-trained ResNet50
        pretrained = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(self.height, self.width, 3)
        )
        pretrained.trainable = False
        
        # Improved architecture
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

    def _create_augmentation_model(self, input_layer):
        """Improved data augmentation pipeline."""
        x = tf.keras.layers.Resizing(self.height, self.width)(input_layer)
        x = tf.keras.layers.RandomBrightness((-0.2, 0.2))(x)
        x = tf.keras.layers.RandomContrast(0.2)(x)
        x = tf.keras.layers.RandomFlip(mode='horizontal')(x)
        x = tf.keras.layers.RandomRotation(0.1)(x)
        x = tf.keras.layers.RandomZoom(0.1)(x)
        x = tf.keras.layers.Lambda(
            tf.keras.applications.resnet.preprocess_input
        )(x)
        
        return tf.keras.models.Model(inputs=input_layer, outputs=x)

    def _f1_score(self, y_true, y_pred):
        """Calculate F1 score metric."""
        true_positives = tf.reduce_sum(y_true * tf.round(y_pred))
        predicted_positives = tf.reduce_sum(tf.round(y_pred))
        actual_positives = tf.reduce_sum(y_true)

        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
        
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return f1

    def _compile_model(self, learning_rate: float = 1e-3):
        """Compile model with improved metrics."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy'),
                self._f1_score
            ]
        )

    def train(self, epochs: int = 40, model_path: str = "thermal_model_resnet.keras"):
        """Train with improved callbacks."""
        if self.class_weights is None:
            self.calculate_class_weights()
            
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
            class_weight=self.class_weights,
            callbacks=callbacks
        )

    def fine_tune(self, num_layers: int = 30):
        """Improved fine-tuning strategy."""
        # Unfreeze the last num_layers layers
        for layer in self.model.layers[2].layers[-num_layers:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
            
        self._compile_model(learning_rate=1e-5)

    def evaluate(self, dataset='test'):
        """Evaluate model with all metrics."""
        eval_ds = {
            'train': self.train_ds,
            'val': self.val_ds,
            'test': self.test_ds
        }.get(dataset)
        
        results = self.model.evaluate(eval_ds)
        metrics = {name: value for name, value in zip(self.model.metrics_names, results)}
        return metrics

    def plot_training_history(self):
        """Plot comprehensive training history."""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
            
        metrics = ['loss', 'accuracy', 'top3_accuracy', '_f1_score']
        plt.figure(figsize=(15, 5))
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, 4, i)
            plt.plot(self.history.history[metric], label=f'Training {metric}')
            plt.plot(self.history.history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'Model {metric}')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.legend()
        
        plt.tight_layout()
        plt.show()

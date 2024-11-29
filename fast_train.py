import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import os
from functools import partial

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    del os.environ['CUDA_VISIBLE_DEVICES']

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"Found {len(physical_devices)} GPUs:")
    for device in physical_devices:
        print(f"  {device}")
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")

def preprocess_image(image, label, image_size, num_classes):
    """Preprocess a single image."""
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 127.5
    image = tf.image.resize(image, image_size, method='bilinear')
    label = tf.one_hot(label, num_classes)
    return image, label

def create_dataset(
    train_dir: str,
    val_dir: str, 
    test_dir: str,
    image_size: Tuple[int, int],
    batch_size: int = 32,
    num_classes: int = 12,
    cache: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create optimized datasets."""
    
    def optimize_dataset(directory: str) -> tf.data.Dataset:
        cache_dir = os.path.join(os.path.dirname(directory), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{os.path.basename(directory)}_cache")
        
        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            image_size=image_size,
            batch_size=None,
            shuffle=True,
            seed=42
        )
        
        preprocess_fn = partial(preprocess_image, image_size=image_size, num_classes=num_classes)
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        if cache:
            ds = ds.cache(cache_path)
        
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return ds
    
    return (
        optimize_dataset(train_dir),
        optimize_dataset(val_dir),
        optimize_dataset(test_dir)
    )

def create_efficient_model(
    input_height: int,
    input_width: int,
    num_classes: int
) -> tf.keras.Model:
    """Create an efficient CNN model."""
    def conv_block(x, filters, kernel_size, strides=1):
        x = tf.keras.layers.SeparableConv2D(
            filters, 
            kernel_size,
            strides=strides,
            padding='same',
            use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.ReLU()(x)
    
    inputs = tf.keras.layers.Input(shape=(input_height, input_width, 3))
    
    x = conv_block(inputs, 32, 3)
    x = conv_block(x, 64, 3, strides=2)
    
    x = conv_block(x, 128, 3)
    x = conv_block(x, 128, 3, strides=2)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    x = tf.keras.layers.Dense(128, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)

def compile_model(
    model: tf.keras.Model,
    initial_learning_rate: float = 1e-3
) -> tf.keras.Model:
    """Compile model with optimized settings."""
    # Define learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate,
        decay_steps=20,  # number of epochs
        alpha=0.1  # minimum learning rate will be 10% of initial rate
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model

def create_callbacks(model_path: str) -> list:
    """Create optimized callbacks."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            model_path + '.weights.h5',
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]

def main():
    # Configuration
    HEIGHT = 48
    WIDTH = int(HEIGHT * (320/240))
    IMAGE_SIZE = (HEIGHT, WIDTH)
    NUM_CLASSES = 12
    EPOCHS = 20
    BATCH_SIZE = 64
    MODEL_PATH = "Efficient_CNN_Model"
    
    train_ds, val_ds, test_ds = create_dataset(
        train_dir='thermal/Dataset-20241124T163610Z-001/Dataset/train',
        val_dir='thermal/Dataset-20241124T163610Z-001/Dataset/val',
        test_dir='thermal/Dataset-20241124T163610Z-001/Dataset/test',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        cache=True
    )
    
    model = create_efficient_model(HEIGHT, WIDTH, NUM_CLASSES)
    model = compile_model(model)
    model.summary()
    
    # Train model
    callbacks = create_callbacks(MODEL_PATH)
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
    )
    
    # Save the final model
    model.save(MODEL_PATH + '_final.keras')
    
    return history, model

if __name__ == "__main__":
    main()

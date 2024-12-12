
import tensorflow as tf
if not tf.config.list_physical_devices('GPU'):
    print("No GPU detected. Training will proceed on CPU.")
else:
    print(f"Using GPU: {tf.config.list_physical_devices('GPU')}")

import tensorflow as tf, numpy as np, matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

train_dir = '/content/drive/MyDrive/Thermal Imaging Project/Greyscale Images/Dataset/train'
val_dir = '/content/drive/MyDrive/Thermal Imaging Project/Greyscale Images/Dataset/val'
test_dir = '/content/drive/MyDrive/Thermal Imaging Project/Greyscale Images/Dataset/test'

height = 75
width = int(height*(320/240))
image_size = (height, width)

# Set up the dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size = image_size,
    batch_size = 32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size = image_size,
    batch_size = 32
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size = image_size,
    batch_size = 32
)

@tf.autograph.experimental.do_not_convert
def grey_to_rgb(image):
  if image.shape[-1] == 1:
    return tf.image.grayscale_to_rgb(image)
  else:
    return image

#Representing the labels using one-hot encoding
train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, 12))).map(lambda x, y: (grey_to_rgb(x), y))
val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, 12))).map(lambda x, y: (grey_to_rgb(x), y))
test_ds = test_ds.map(lambda x, y: (x, tf.one_hot(y, 12))).map(lambda x, y: (grey_to_rgb(x), y))

train_ds = train_ds.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

input_layer = tf.keras.layers.Input(shape = (height, width, 3))
resizing_layer = tf.keras.layers.Resizing(height, width)(input_layer)
brightness_adjustment_layer = tf.keras.layers.RandomBrightness((-0.5, 0.5))(resizing_layer)
contrast_adjustment_layer = tf.keras.layers.RandomContrast(0.9)(brightness_adjustment_layer)
flipping_layer = tf.keras.layers.RandomFlip()(contrast_adjustment_layer)
preprocessing_layer = tf.keras.layers.Lambda(tf.keras.applications.inception_v3.preprocess_input)(flipping_layer)

data_augmentation_model = tf.keras.models.Model(inputs = input_layer, outputs = preprocessing_layer)

pretrained_model = tf.keras.applications.InceptionV3(
    include_top = False,
    weights = 'imagenet'
    )

for layer in pretrained_model.layers:
  layer.trainable = False

pretrained_model.summary()

input_layer = tf.keras.layers.Input(shape = data_augmentation_model.input_shape[1:])

output_aug = data_augmentation_model(input_layer)
output_pretrained = pretrained_model(output_aug)

p_plus_a = tf.keras.models.Model(inputs = input_layer, outputs = output_pretrained)

avg = tf.keras.layers.GlobalAveragePooling2D()(p_plus_a.output)
output = tf.keras.layers.Dense(12, activation = 'softmax')(avg)
model = tf.keras.models.Model(inputs = p_plus_a.input, outputs = output)

optimizer = tf.keras.optimizers.Nadam(learning_rate = 1e-3)
loss = tf.keras.losses.CategoricalCrossentropy(name = "val_loss")
accuracy = tf.keras.metrics.Accuracy()

model.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = [accuracy]
    )

model_checkpoint = tf.keras.callbacks.ModelCheckpoint("/content/drive/MyDrive/Thermal Imaging Project/Greyscale Images/Transfer Learning Model Using InceptionV3.keras", save_best_only = True, monitor = "val_binary_accuracy")
model_rop_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = "loss", factor = 0.1, patience = 10, verbose = 1)

model.summary()

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 40,
    callbacks = [model_checkpoint, model_rop_lr]
)

model = tf.keras.models.load_model("/content/drive/MyDrive/Thermal Imaging Project/Greyscale Images/Transfer Learning Model Using InceptionV3.keras", custom_objects={'preprocess_input': tf.keras.applications.inception_v3.preprocess_input})

model.evaluate(test_ds)

model.layers[2].layers[-19].trainable = True

model.layers[2].layers[-27].trainable = True

model.summary()

optimizer = tf.keras.optimizers.Nadam(learning_rate = 1e-3)
loss = tf.keras.losses.CategoricalCrossentropy(name = "val_loss")
accuracy = tf.keras.metrics.Accuracy()

model.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = [accuracy]
    )

model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 40,
    callbacks = [model_checkpoint, model_rop_lr]
)

model.save("/content/drive/MyDrive/Thermal Imaging Project/Greyscale Images/Transfer Learning Model Using InceptionV3.keras")


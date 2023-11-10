
import tensorflow as tf
from tensorflow import keras
import keras_cv

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = keras_cv.models.ResNet50V2Backbone()

image_size = (224, 224)
batch_size = 64

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    "treat_images/Train_Images",
    validation_split=0.2,
    subset="both",
    label_mode="categorical",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


epochs = 5

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

retina_net = keras_cv.models.FasterRCNN(
    classes=3,
    bounding_box_format="xyxy",
    backbone=model
)

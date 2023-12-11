import os
import pathlib
import tensorflow as tf

from retinaNet.utils.BoundingBoxUtilities import convert_to_xywh

autotune = tf.data.AUTOTUNE

def process_path(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, dtype=tf.uint8)
    
    bbox = tf.constant([[0.1111, 0.1111, 0.1111, 0.1111],
                        [0.1111, 0.1111, 0.1111, 0.1111],
                        [0.1111, 0.1111, 0.1111, 0.1111],
                        [0.1111, 0.1111, 0.1111, 0.1111]], dtype=tf.float32)

    bbox = tf.stack(
        [
            bbox[:, 0] * 640,
            bbox[:, 1] * 640,
            bbox[:, 2] * 640,
            bbox[:, 3] * 640,
        ],
        axis=-1,
    )
    
    bbox = convert_to_xywh(bbox)
    img_type = tf.constant([0,1,2,3], dtype=tf.int32)
    return image, bbox, img_type

def createTrainingDataset(batch_size, label_encoder):
    pathTraining = os.environ["trainingPath"]
    trainDataset = tf.data.Dataset.list_files(str(pathlib.Path(pathTraining + "*.png")))
    
    trainDataset = trainDataset.map(process_path, num_parallel_calls=autotune)
    trainDataset = trainDataset.shuffle(4 * batch_size)
    
    trainDataset = trainDataset.padded_batch(
        batch_size=batch_size, drop_remainder=True
    )
    
    trainDataset = trainDataset.map(
        label_encoder.encode_batch, num_parallel_calls=autotune
    )
    
    trainDataset = trainDataset.ignore_errors()
    trainDataset = trainDataset.prefetch(autotune)

    return trainDataset


def createValidationDataset(batch_size, label_encoder):
    pathValidation = os.environ["validationPath"]
    validationDataset = tf.data.Dataset.list_files(str(pathlib.Path(pathValidation + "*.png")))
    
    validationDataset = validationDataset.map(process_path, num_parallel_calls=autotune)
    valDataset = validationDataset.shuffle(4 * batch_size)
    
    valDataset = valDataset.padded_batch(
        batch_size=1, drop_remainder=True
    )
    
    valDataset = valDataset.map(
        label_encoder.encode_batch, num_parallel_calls=autotune
    )
    
    valDataset = valDataset.ignore_errors()
    valDataset = valDataset.prefetch(autotune)

    return valDataset
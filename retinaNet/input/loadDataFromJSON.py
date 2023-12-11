import os
import tensorflow as tf
import json
from functools import partial

from retinaNet.utils.BoundingBoxUtilities import convert_to_xywh
from retinaNet.input.externalLibDataAugmentation import prepareForAugment, prepareForAugmentVal, setShapes

os.environ["jsonPath"] = "datasetFaturas/datasetInfo.json"

autotune = tf.data.AUTOTUNE

def dictToDataset(datasetInfo, key, total):
    filePaths = []
    finalLabels = []

    count = 0

    for i, invoice in enumerate(datasetInfo[key]):
        itemLabels = []
        filePaths.append(invoice)
        for label, bbox in datasetInfo[key][invoice].items():
            bbox.append(float(label))
            itemLabels.append(bbox)

        count +=1
        finalLabels.append(itemLabels)
    
    return tf.data.Dataset.from_tensor_slices((filePaths, tf.ragged.constant(finalLabels)))

def processFiles(imageFile, labelBbox):
    image = tf.io.read_file(imageFile)
    image = tf.image.decode_png(image, dtype=tf.uint8)

    labelBbox = labelBbox.to_tensor()
    
    labels =  tf.stack(labelBbox[:, 4])
    labels = tf.cast(labels, dtype=tf.int32)

    bbox = tf.stack(labelBbox[:, 0:4])
    bbox = tf.cast(bbox, dtype=tf.float32)

    return image, bbox, labels

def processFilesEval(imageFile, labelBbox):
    image = tf.io.read_file(imageFile)
    image = tf.image.decode_png(image, dtype=tf.uint8)

    labelBbox = labelBbox.to_tensor()
    
    labels =  tf.stack(labelBbox[:, 4])
    labels = tf.cast(labels, dtype=tf.int32)

    bbox = tf.stack(labelBbox[:, 0:4])
    bbox = tf.cast(bbox, dtype=tf.float32)
    
    bbox = tf.stack([
        bbox[:, 0] * 640,
        bbox[:, 1] * 640,
        bbox[:, 2] * 640,
        bbox[:, 3] * 640], axis=-1)

    return image, bbox, labels

def centralizeBBox(image, bbox, label):
    bbox = tf.stack([
        bbox[:, 0] * 640,
        bbox[:, 1] * 640,
        bbox[:, 2] * 640,
        bbox[:, 3] * 640], axis=-1)
    bbox = convert_to_xywh(bbox)
    return image, bbox, label

def createTrainingDataset(batch_size, label_encoder):
    f = open(os.environ["jsonPath"]) 
    datasetInfo = json.load(f)
    trainDataset = dictToDataset(datasetInfo, "trainInvoices", 500)
    trainDataset = trainDataset.map(processFiles, num_parallel_calls=autotune)
    trainDataset = trainDataset.map(partial(prepareForAugment), num_parallel_calls=autotune)
    trainDataset = trainDataset.map(setShapes, num_parallel_calls=autotune)
    trainDataset = trainDataset.map(centralizeBBox, num_parallel_calls=autotune)
    
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
    f = open(os.environ["jsonPath"]) 
    datasetInfo = json.load(f)
    validationDataset = dictToDataset(datasetInfo, "valInvoices", 75)
    validationDataset = validationDataset.map(processFiles, num_parallel_calls=autotune)
    # validationDataset = validationDataset.map(partial(prepareForAugmentVal), num_parallel_calls=autotune)
    # validationDataset = validationDataset.map(setShapes, num_parallel_calls=autotune)
    validationDataset = validationDataset.map(centralizeBBox, num_parallel_calls=autotune)
    valDataset = validationDataset.shuffle(4 * batch_size)
    
    valDataset = valDataset.padded_batch(
        batch_size=batch_size, drop_remainder=True
    )
    
    valDataset = valDataset.map(
        label_encoder.encode_batch, num_parallel_calls=autotune
    )
    
    valDataset = valDataset.ignore_errors()
    valDataset = valDataset.prefetch(autotune)

    return valDataset

def createEvaluationDataset():
    f = open(os.environ["jsonPath"]) 
    datasetInfo = json.load(f)
    evalDataset = dictToDataset(datasetInfo, "valInvoices", 75)
    evalDataset = evalDataset.map(processFilesEval, num_parallel_calls=autotune)
    return evalDataset
import os
import pathlib
import tensorflow as tf

from retinaNet.components.EncodeLabels import LabelEncoder
from retinaNet.components.RetinaLoss import RetinaNetLoss
from retinaNet.components.RetinaNet import RetinaNet
from retinaNet.components.DecodePredictions import DecodePredictions
from retinaNet.CNN.resNet50 import ResNet50Reduced
from retinaNet.utils.BoundingBoxUtilities import convert_to_xywh, visualize_detections


"""
## Setting up training parameters
"""

model_dir = "retinaNet/weightsTraining/"
label_encoder = LabelEncoder()

num_classes = 4
batch_size = 8

learning_rates = [2.5e-07, 0.0000625, 0.000125, 0.0000625, 0.00001, 2.5e-07]
learning_rate_boundaries = [250, 750, 2500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

"""
## Initializing and compiling model
"""

resnet50_backbone = ResNet50Reduced(include_top=False, weights=None, input_shape=[None, None, 3])
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)

"""
## Setting up callbacks
"""

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
]

def process_path(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, dtype=tf.uint8)
    bbox = tf.constant([[0.19103, 0.07022, 0.49576, 0.13823], 
                        [0.65652, 0.06845, 0.80579, 0.10201],
                        [0.45690, 0.31753, 0.80756, 0.75652],
                        [0.45425, 0.79361, 0.81286, 0.85986]], dtype=tf.float32)
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
    img_type = tf.constant([0, 1, 2, 3], dtype=tf.int32)
    return image, bbox, img_type

trainDataset = tf.data.Dataset.list_files(str(pathlib.Path("datasetFaturas/MV_COMPLETE_TRAIN/Pag1/"+"*.png")))

autotune = tf.data.AUTOTUNE
trainingDataset = trainDataset.map(process_path, num_parallel_calls=autotune)
trainDataset = trainingDataset.shuffle(4 * batch_size)
trainDataset = trainDataset.padded_batch(
    batch_size=batch_size, drop_remainder=True
)
trainDataset = trainDataset.map(
    label_encoder.encode_batch, num_parallel_calls=autotune
)
trainDataset = trainDataset.ignore_errors()
trainDataset = trainDataset.prefetch(autotune)


validationDataset = tf.data.Dataset.list_files(str(pathlib.Path("datasetFaturas//MV_COMPLETE_VALIDATION/Pag1/"+"*.png")))

autotune = tf.data.AUTOTUNE
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


"""
## Training the model
"""
epochs = 35
model.fit(
    trainDataset,
    validation_data=valDataset,
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)

"""
## Loading weights
"""
# Change this to `model_dir` when not using the downloaded weights
weights_dir = "retinaNet/weightsTraining"

latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)


"""
## Building inference model
"""
image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.90)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

"""
## Generating detections
"""
def prepare_image(image):
    #image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), tf.constant([1])


"""
## Get test examples
"""
test_dataset = tf.data.Dataset.list_files(str(pathlib.Path("datasetFaturas/MV_COMPLETE_VALIDATION/Pag1/"+"*.png")))
test_dataset = test_dataset.map(process_path)

"""
## Visualize test examples and detections made by the network
"""
for sample in test_dataset.take(5):
    image = sample[0]
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    import ipdb; ipdb.set_trace()

    classes = {0: "Endereço", 1: "Instalação", 2: "Faturamento", 3: "Impostos"}
    
    class_names = [classes[x] for x in detections.nmsed_classes[0][:num_detections]]
    
    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )

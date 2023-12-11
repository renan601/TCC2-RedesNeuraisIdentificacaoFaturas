import os
import tensorflow as tf
import json
from dotenv import load_dotenv

from retinaNet.components.EncodeLabels import LabelEncoder
from retinaNet.components.RetinaLoss import RetinaNetLoss
from retinaNet.components.RetinaNet import RetinaNet
from retinaNet.components.DecodePredictions import DecodePredictions

from retinaNet.CNN.resNet50 import ResNet50Reduced

from retinaNet.input.loadDataFromJSON import createTrainingDataset, createValidationDataset, createEvaluationDataset

from retinaNet.utils.BoundingBoxUtilities import visualize_detections
from retinaNet.metrics.objectDetectionMetrics import metricsPrepare

# Enable to consume a personalized value of video board memory
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.set_logical_device_configuration(physical_devices[0],[tf.config.LogicalDeviceConfiguration(memory_limit=12000)])
except:
    pass

"""
## Setting up training parameters
"""
model_dir = "retinaNet/weightsTrainingTeste/"
label_encoder = LabelEncoder()

num_classes = 5
batch_size = 16


load_dotenv()
f = open(os.environ["JSON_PATH"]) 
datasetInfo = json.load(f)
classesJSON = datasetInfo["labelNames"]

classes = {}
for item in classesJSON.keys():
    classes[int(item)] = classesJSON[item]

"""
## Initializing and compiling model
"""
resnet50_backbone = ResNet50Reduced(input_shape=[640, 640, 3])
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
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
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        min_delta=0.0001,
        patience=3,
        cooldown=1,
        factor=0.4,
        verbose=1,
        min_lr=6e-8
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0100,
        patience=8,
        verbose=1
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir="logsCenarioTeste",
        histogram_freq=10,
        write_images=True,
        write_steps_per_second=True,
        update_freq="epoch",
        embeddings_freq=10
    )
]

trainDataset = createTrainingDataset(batch_size, label_encoder)
valDataset = createValidationDataset(batch_size, label_encoder)

"""
## Training the model
"""
epochs = 30
model.fit(
    trainDataset,
    validation_data=valDataset,
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1
)

"""
## Loading weights
"""
weights_dir = "retinaNet/weightsTrainingTeste/"

latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)

"""
## Get test examples
"""
evalDataset = createEvaluationDataset()

        
"""
## Building inference model
"""
image = tf.keras.Input(shape=[640, 640, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.5, num_classes=num_classes)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

"""
## Generating detections
"""
def prepare_image(image):
    return tf.expand_dims(image, axis=0), tf.constant([1.0], dtype=tf.float32)

"""
## Prepare metrics evaluation
"""
TP = {}
FP = {}
FN = {}

for item in classes.keys():
    TP[item] = 0
    FP[item] = 0
    FN[item] = 0

"""
## Visualize test examples and detections made by the network
"""
for sample in evalDataset.take(10):
    image = sample[0]
    gtBbox = sample[1]
    gtClasses = sample[2]

    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    
    metricsPrepare(detections.nmsed_boxes[0][:num_detections], 
                    detections.nmsed_classes[0][:num_detections], 
                    gtBbox, gtClasses, classes, TP, FP, FN)
    
    predClassNames = [classes[x] for x in detections.nmsed_classes[0][:num_detections]]
    gtClassNames = [classes[x] for x in gtClasses.numpy()]

    print(f"TP for objects: {TP}")
    print(f"FP for objects: {FP}")
    print(f"FN for objects: {FN}")

    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        predClassNames,
        detections.nmsed_scores[0][:num_detections])

    visualize_detections(
        image,
        gtBbox,
        gtClassNames,
        tf.range(len(gtClasses), delta=1, dtype=tf.int32),
        color=[1, 0, 0])

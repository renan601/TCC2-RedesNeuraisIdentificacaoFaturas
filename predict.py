import tensorflow as tf
import os
import json

from PIL import Image
from os import listdir
from os.path import isfile, join
from dotenv import load_dotenv

from retinaNet.components.EncodeLabels import LabelEncoder
from retinaNet.components.RetinaLoss import RetinaNetLoss
from retinaNet.components.RetinaNet import RetinaNet
from retinaNet.components.DecodePredictions import DecodePredictions
from retinaNet.utils.BoundingBoxUtilities import visualize_detections, make_square
from retinaNet.CNN.resNet50 import ResNet50Reduced

"""
## Setting up training parameters
"""
label_encoder = LabelEncoder()

num_classes = 5
batch_size = 16

"""
## Initializing and compiling model
"""

resnet50_backbone = ResNet50Reduced(include_top=False, weights=None, input_shape=[640, 640, 3])
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-7)
model.compile(loss=loss_fn, optimizer=optimizer)

"""
## Loading weights
"""
weights_dir = "retinaNet/weightsTrainingCenario3Ciclo1/"

latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)


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

load_dotenv()
f = open(os.environ["JSON_PATH"]) 
datasetInfo = json.load(f)
classesJSON = datasetInfo["labelNames"]

predictPath = "datasetFaturas/PREDICT/"
onlyfiles = [f for f in listdir(predictPath) if isfile(join(predictPath, f))]

classes = {}
for item in classesJSON.keys():
    classes[int(item)] = classesJSON[item]
    if not os.path.isdir(predictPath + classesJSON[item]):
        os.mkdir(predictPath + classesJSON[item])

for imageFile in onlyfiles:
    image = Image.open(predictPath + imageFile)
    width, height = image.size

    max_side_size = 640
    image.thumbnail((max_side_size, max_side_size))

    image, width_reduction, height_reduction = make_square(image, max_side_size)
    imageTF  = tf.keras.utils.img_to_array(image)

    input_image, ratio = prepare_image(imageTF)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    
    predClassNames = [classes[x] for x in detections.nmsed_classes[0][:num_detections]]
    predBbox = detections.nmsed_boxes[0][:num_detections]

    if height > width:
        proportion = round(height / image.size[0], 3)
        for bbox in predBbox:
            bbox[0] = (bbox[0] - width_reduction/2) * proportion
            bbox[1] = bbox[1] * proportion
            bbox[2] = (bbox[2] - width_reduction/2) * proportion
            bbox[3] = bbox[3] * proportion
    else:
        proportion = round(width / image.size[0], 3)
        for bbox in predBbox:
            bbox[0] = (bbox[0] - height_reduction/2) * proportion
            bbox[1] = bbox[1] * proportion
            bbox[2] = (bbox[2] - height_reduction/2) * proportion
            bbox[3] = bbox[3] * proportion

    originalImage = Image.open(predictPath + imageFile)
    print(f"\nImage: {imageFile}")

    for i, pred in enumerate(predClassNames):
        bbox = (predBbox[i][0], predBbox[i][1], predBbox[i][2], predBbox[i][3])
        print(f"Objeto: {pred} - Bbox (xmin,ymin,xmax,ymax): {bbox}")
        predCrop = originalImage.crop(bbox)
        predCrop.save(f"{predictPath}{pred}/{i}-{imageFile}")

    originalImage = tf.keras.utils.img_to_array(originalImage)
    visualize_detections(
        originalImage,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        predClassNames,
        detections.nmsed_scores[0][:num_detections])
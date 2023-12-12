import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

from retinaNet.components.EncodeLabels import LabelEncoder
from retinaNet.components.RetinaLoss import RetinaNetLoss
from retinaNet.components.RetinaNet import RetinaNet
from retinaNet.components.DecodePredictions import DecodePredictions

from retinaNet.CNN.resNet50 import ResNet50Reduced

from retinaNet.input.loadDataFromJSON import createEvaluationDataset

from retinaNet.metrics.objectDetectionMetrics import metricsPrepareEvaluate

"""
## Setting up training parameters
"""
label_encoder = LabelEncoder()

num_classes = 5
batch_size = 16

"""
## Initializing and compiling model
"""

resnet50_backbone = ResNet50Reduced(input_shape=[640, 640, 3])
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
## Get test examples
"""
evalDataset = createEvaluationDataset()

"""
## Get metrics
"""
confThresholds = np.arange(1, 0.29, -0.01)

classes = {0: "Endereço", 1: "Instalação", 2: "Faturamento", 3: "Impostos", 4: "Leitura"}

recallByConf = {}
precisionByConf = {}

meanRecall = []
meanPrecision = []

metricsByConf = {}

for conf in confThresholds:
    conf = round(conf, 2)

    metricsByConf[conf] = {"TP": [], "FP": [], "FN": []}

    for conf in confThresholds:
        conf = round(conf, 2)

        metricsByConf[conf] = {"TP": [], "FP": [], "FN": []}

        for classNumber in classes.keys():
            metricsByConf[conf]["TP"].append(0)
            metricsByConf[conf]["FP"].append(0)
            metricsByConf[conf]["FN"].append(0)
    
"""
## Building inference model
"""
image = tf.keras.Input(shape=[640, 640, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.3, num_classes=num_classes)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

"""
## Generating detections
"""
def prepare_image(image):
    return tf.expand_dims(image, axis=0), tf.constant([1.0], dtype=tf.float32)

"""
## Prepare metrics evaluation
"""
for sample in evalDataset:
    image = sample[0]
    gtBbox = sample[1]
    gtClasses = sample[2]

    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    
    metricsPrepareEvaluate(detections.nmsed_boxes[0][:num_detections], 
                    detections.nmsed_classes[0][:num_detections], 
                    gtBbox, gtClasses, classes,
                    detections.nmsed_scores[0][:num_detections],
                    metricsByConf)


## Calculate AP using TP, FP and FN in a set off confidence values
AP = []
for object in classes.keys():
    recallList = []
    precisionList = []
    
    for conf in metricsByConf.keys():
        TP = metricsByConf[conf]["TP"][object]
        FN = metricsByConf[conf]["FN"][object]
        FP = metricsByConf[conf]["FP"][object]
        
        if TP != 0:
            recallList.append( round( TP / (TP + FN), 2) )
            precisionList.append( round( TP / (TP + FP), 2) )

    if conf == 0.5:
        print(f"Conf 0.5 - Recall: {round( TP / (TP + FN), 2)} for {classes[object]}")
        print(f"Conf 0.5 - Precision: {round( TP / (TP + FP), 2)} for {classes[object]} \n")

    if recallList[0] > 0.05:
        recallList.insert(0, 0.05)
        precisionList.insert(0, 1)
    
    AP.append(auc(recallList, precisionList))
        
    # Plotting the recall-precision curve
    plt.plot(np.array(recallList), np.array(precisionList))
    plt.title(f"Recall by Precision Curve For {classes[object]}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

for i, object in enumerate(classes.keys()):
    print(f"{classes[i]}: {AP[i]}")

mAP = sum(AP) / len(classes)
print(f"mAP: {mAP}")
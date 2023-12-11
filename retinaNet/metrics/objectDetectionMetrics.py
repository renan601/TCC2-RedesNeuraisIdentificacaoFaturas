import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import auc
import copy


from retinaNet.utils.BoundingBoxUtilities import compute_iou

def metricsPrepare(predBoxes, predClasses, gtBoxes, gtClasses, classNames, TP, FP, FN):
    gtBoxesXYWH = np.copy(gtBoxes)
    predBoxesXYWH = np.copy(predBoxes)

    predClasses = predClasses.astype(dtype=np.int16)
    gtClasses = gtClasses.numpy()
    
    predItems = {}
    for item in classNames.keys():
        predItems[item] = []

    for i, bbox in enumerate(predBoxesXYWH):
        predItems[predClasses[i]].append(i)
        bbox[0] = round(bbox[0], 2)
        bbox[1] = round(bbox[1], 2)
        bbox[2] = round(bbox[2] - bbox[0], 2)
        bbox[3] = round(bbox[3] - bbox[1], 2)

    for i, bbox in enumerate(gtBoxesXYWH):
        bbox[0] = round(bbox[0], 2)
        bbox[1] = round(bbox[1], 2)
        bbox[2] = round(bbox[2] - bbox[0], 2)
        bbox[3] = round(bbox[3] - bbox[1], 2)

    IoU = compute_iou(predBoxesXYWH, gtBoxesXYWH)

    computeMetricsForRecallPrecision(IoU, predItems, gtClasses, TP, FP, FN)

def metricsPrepareEvaluate(predBoxes, predClasses, gtBoxes, gtClasses, classNames, scores, metricsByConf):
    gtBoxesXYWH = np.copy(gtBoxes)
    predBoxesXYWH = np.copy(predBoxes)

    predClasses = predClasses.astype(dtype=np.int16)
    gtClasses = gtClasses.numpy()
    
    predItems = {}
    for item in classNames.keys():
        predItems[item] = []

    for i, bbox in enumerate(predBoxesXYWH):
        predItems[predClasses[i]].append(i)
        bbox[0] = round(bbox[0], 2)
        bbox[1] = round(bbox[1], 2)
        bbox[2] = round(bbox[2] - bbox[0], 2)
        bbox[3] = round(bbox[3] - bbox[1], 2)

    for i, bbox in enumerate(gtBoxesXYWH):
        bbox[0] = round(bbox[0], 2)
        bbox[1] = round(bbox[1], 2)
        bbox[2] = round(bbox[2] - bbox[0], 2)
        bbox[3] = round(bbox[3] - bbox[1], 2)

    IoU = compute_iou(predBoxesXYWH, gtBoxesXYWH)
    computeMetricsForRecallPrecisionByConf(IoU, predItems, gtClasses, scores, metricsByConf)


def calculateRecallPrecision(recall, precision, TP, FN, FP):
    """Calculate Mean Average Precision"""
    meanRecall = 0
    meanPrecision = 0

    for item in TP.keys():
        try:
            recall[item] = round((TP[item] / (TP[item] + FN[item])), 3)
        except:
            recall[item] = 0
        try:
            precision[item] = round(TP[item] / (TP[item] + FP[item]), 3)
        except:
            precision[item] = 0

        meanRecall += recall[item]
        meanPrecision += precision[item]

    meanRecall = meanRecall / len(recall)
    meanPrecision = meanPrecision / len(precision)

    return meanRecall, meanPrecision


def computeMetricsForRecallPrecision(IoU, predItems, gtClasses, TP, FP, FN):
    for pos, item in enumerate(gtClasses):
        # Object exists and it was not predicted (False Negative)
        if predItems[item] == []:
            FN[item] += 1
        
        # Object exists and received one prediction with the right label
        elif len(predItems[item]) == 1:
            # Match IoU
            if IoU[predItems[item]][pos] > 0.75:
                TP[item] += 1
            # Not match IoU
            else:
                FP[item] += 1
                FN[item] += 1
        
        # Object exists and it received several predictions with the right label
        else:
            correctPredition = False
            for predObj in predItems[item]:
                if IoU[predObj][pos] > 0.75:
                    correctPredition = True
                    break
            
            # Object exists and it received several predictions, one was right, the others don't
            if correctPredition:
                TP[item] += 1
                FP[item] += len(predItems[item]) - 1
            # Object exists and it received several predictions, none was right
            else:
                FN[item] += 1
                FP[item] += len(predItems[item])
            
        
        predItems.pop(item)

    
    # Objects predicted were not in ground truth
    popItems = []
    for item in predItems.keys():
        FP[item] += len(predItems[item])
        popItems.append(item)
    
    for item in popItems:
        predItems.pop(item)


def computeMetricsForRecallPrecisionByConf(IoU, predItemsOriginal, gtClasses, scores, metricsByConf):
    scores = scores.tolist()
    for i, score in enumerate(scores):
        scores[i] = round(score, 2)

    confThresholds = np.arange(1, 0.29, -0.01)
    for conf in confThresholds:
        conf = round(conf, 2)
        predItems = copy.deepcopy(predItemsOriginal)
        
        for pos, item in enumerate(gtClasses):
            # Object exists and it was not predicted (False Negative)
            if predItems[item] == []:
                metricsByConf[conf]["FN"][item] += 1
            
            # Object exists and received one prediction with the right label
            elif len(predItems[item]) == 1 and scores[predItems[item][0]] >= conf:
                # Match IoU
                if IoU[predItems[item]][pos] > 0.75:
                    metricsByConf[conf]["TP"][item] += 1
                # Not match IoU
                else:
                    metricsByConf[conf]["FP"][item] += 1
                    metricsByConf[conf]["FN"][item] += 1
            
            # Object exists and it received several predictions with the right label
            elif len(predItems[item]) > 1:
                correctPredition = False

                auxPredItems = copy.deepcopy(predItems[item])
                for predObj in auxPredItems:
                    if scores[predObj] < conf:
                        predItems[item].remove(predObj)
                        

                for predObj in predItems[item]:
                    if IoU[predObj][pos] > 0.75:
                        correctPredition = True
                        break
                
                # Object exists and it received several predictions, one was right, the others don't
                if correctPredition:
                    metricsByConf[conf]["TP"][item] += 1
                    metricsByConf[conf]["FP"][item] += len(predItems[item]) - 1
                
                # Object exists and it received several predictions, none was right
                else:
                    metricsByConf[conf]["FN"][item] += 1
                    metricsByConf[conf]["FP"][item] += len(predItems[item])
            
            else:
                metricsByConf[conf]["FN"][item] += 1
            
            predItems.pop(item)

        
        # Objects predicted were not in ground truth
        popItems = []
        for item in predItems.keys():
            metricsByConf[conf]["FP"][item] += len(predItems[item])
            popItems.append(item)
        
        for item in popItems:
            predItems.pop(item)
                    

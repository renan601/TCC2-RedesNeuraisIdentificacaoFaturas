import albumentations as A
import tensorflow as tf

"""
## List of augmentations to apply
"""
transformTrain = A.Compose([
        A.BBoxSafeRandomCrop(p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
        A.HueSaturationValue(p=0.4),
        A.PadIfNeeded(min_height=640, min_width=640),
    ], bbox_params=A.BboxParams(format='albumentations', min_visibility=0.95, label_fields=['class_labels']), p=1)

transformVal = A.Compose([
        A.BBoxSafeRandomCrop(p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
        A.HueSaturationValue(p=0.4),
        A.PadIfNeeded(min_height=640, min_width=640),
    ], bbox_params=A.BboxParams(format='albumentations', min_visibility=0.95, label_fields=['class_labels']), p=1)


"""
## Apply augmentation to dataset
"""
def augmentWithAlbumentations(image, bbox, label, dataset):
    if dataset == "val":
        transformed = transformVal(image=image, bboxes=bbox, class_labels=label)
    else:
        transformed = transformTrain(image=image, bboxes=bbox, class_labels=label)

    transformed_image = tf.cast(transformed['image'], dtype=tf.uint8)
    transformed_bboxes = tf.cast(transformed['bboxes'], dtype=tf.float32)
    transformed_class_labels = tf.cast(transformed['class_labels'], dtype=tf.int32)
    return transformed_image, transformed_bboxes, transformed_class_labels

"""
## Prepare for augmentation process
"""
def prepareForAugment(image, bbox, label, augmentWithAlbumentations=augmentWithAlbumentations):
    image, bbox, label = tf.numpy_function(func=augmentWithAlbumentations, inp=[image, bbox, label, "train"], Tout=[tf.uint8, tf.float32, tf.int32])
    return image, bbox, label

def prepareForAugmentVal(image, bbox, label, augmentWithAlbumentations=augmentWithAlbumentations):
    image, bbox, label = tf.numpy_function(func=augmentWithAlbumentations, inp=[image, bbox, label, "val"], Tout=[tf.uint8, tf.float32, tf.int32])
    return image, bbox, label


"""
## Reshape dataset after augmentation
"""
def setShapes(img, bbox, label, img_shape=(640,640,3), bbox_shape=(None,None), label_shape=(None,)):
    img.set_shape(img_shape)
    bbox.set_shape(bbox_shape)
    label.set_shape(label_shape)
    return img, bbox, label

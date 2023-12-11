from pdf2image import convert_from_path
import os
from os import listdir
from os.path import isfile, join
from PIL import Image


def make_square(im, min_size, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im, size - x, size - y


"""
## Resize big images in 640x640 PNG images with padding when needed and print bbox coordinates relative to height
"""
directory = 'datasetFaturas/PDFs/CEMIGLV3/'
img_count = 0
max_side_size = 640
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        # Open image with PIL
        image = Image.open(f)
        width, height = image.size

        # Coordinates for objects in the specific layout
        all_labels = {
            "labelObj1": [99.999, 99.999, 99.999, 99.999],
            "labelObj2": [99.999, 99.999, 99.999, 99.999],
            "labelObj3": [99.999, 99.999, 99.999, 99.999],
            "labelObj4": [99.999, 99.999, 99.999, 99.999]
        }

        # Resize keeping aspect ratio
        image.thumbnail((max_side_size, max_side_size))

        # Pad image and get pad size
        image, width_reduction, height_reduction = make_square(image, max_side_size)
        
        # Find new bbox value with resized and padded image. Divide by max_side_size to get relative values for X and Y
        if height > width:
            scale_factor = height/max_side_size
            for keys in all_labels.keys():
                all_labels[keys][0] = ((all_labels[keys][0] / scale_factor) + width_reduction / 2) / max_side_size
                all_labels[keys][1] = (all_labels[keys][1] / scale_factor) / max_side_size
                all_labels[keys][2] = ((all_labels[keys][2] / scale_factor) + width_reduction / 2) / max_side_size
                all_labels[keys][3] = (all_labels[keys][3] / scale_factor) / max_side_size
        else:
            scale_factor = width/max_side_size
            for keys in all_labels.keys():
                all_labels[keys][0] = ((all_labels[keys][0] / scale_factor) + height_reduction / 2) / max_side_size
                all_labels[keys][1] = (all_labels[keys][1] / scale_factor) / max_side_size
                all_labels[keys][2] = ((all_labels[keys][2] / scale_factor) + height_reduction / 2) / max_side_size
                all_labels[keys][3] = (all_labels[keys][3] / scale_factor) / max_side_size
        
        print(all_labels)
        
        # Save resized images
        image.save(f'datasetFaturas/ModeloX/{img_count}.png', 'png')
        
    img_count +=1
import tensorflow as tf
import argparse
from models.EDet import EDet
from time import time, sleep
import cv2
import numpy as np
from preprocessing import *
from generators.CocoGenerator import CocoGenerator

# VARIABLES
model_name = 'iou_debug'
img_size = (224, 224)
box_delta = 100


colors = np.random.uniform(0, 1, size=(80, 3))  # random colors for classes
coco = CocoGenerator(data_dir=r'/home/varuzh/my/coco/2017', set_name="val2017")
classes = coco.labels

print(classes)
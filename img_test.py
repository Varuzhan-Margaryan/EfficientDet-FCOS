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


########################

# FUNCTIONS
def network(model, img):
    img = img.copy()
    img = cv2.resize(img, img_size)
    if verbose > 1:
        print('I shape of input image:\t', img.shape)

    X = tf.expand_dims(img, 0)
    class_map, box_map = model.forward_pass(X, training=False)
    class_map, box_map = class_map[0].numpy(), box_map[0].numpy()
    if verbose > 1:
        print('I shape of class map:\t', class_map.shape)
        print('I shape of box map:\t', box_map.shape)
        print('I max min box map:\t', box_map.max(), box_map.min())
        print('I max min class map:\t', class_map.max(), class_map.min())
    return class_map, box_map


def show_boxes(img, boxes_classes, write_labels=True):
    img = img.copy()
    img = cv2.resize(img, img_size)
    draw_boxes(img, boxes_classes, colors, classes, write_labels)
    img = img[:, :, ::-1]
    img = cv2.resize(img, (500, 500))
    cv2.imshow('boxes', img)


def show_seg(img, class_map, colors, classes):
    img = img.copy()
    img = cv2.resize(img, (112, 112))
    show_segments(img, class_map, colors, classes)
    img = img[:, :, ::-1]
    img = cv2.resize(img, (500, 500))
    cv2.imshow('seg', img)


########################

# MODEL
parser = argparse.ArgumentParser(description="Global parameters for testing.")
parser.add_argument("--model_name", type=str, default=model_name, help="name of model")
parser.add_argument("--verbose", type=int, default=10, help="verbosity of model")
parser.add_argument("--threshold", type=float, default=0.5, help="verbosity of model")
parser.add_argument('--seg', dest='seg', action='store_true')
parser.set_defaults(seg=False)
args = parser.parse_args()

model = EDet(
    img_size=img_size,
    FPN_depth=1,
    FPN_conv_filters=8,
    clsbox_conv_filters=8,
    classnet_convs_num=1,
    boxnet_convs_num=1,
    training=False,
    model_name=args.model_name
)
model.ckpt_and_sum_setup()
##########################

verbose = args.verbose
colors = np.random.uniform(0, 1, size=(80, 3))  # random colors for classes
coco = CocoGenerator(data_dir=r'C:\Users\Varuzhan\Desktop\Code\Python\PyCharm\EDet\data', set_name="val2017")
classes = coco.labels

while True:
    img = coco.load_rand_image() / 255.0
    if verbose > 1:
        print('I initial shape of image:\t', img.shape)

    class_map, box_map = network(model, img)

    boxes_classes = get_resized_boxes_and_classes(class_map, box_map, img_size,
                                                  threshold=args.threshold, box_delta=box_delta)

    show_boxes(img, boxes_classes, write_labels=True)
    show_seg(img, class_map, colors, classes)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# class_pred, box_pred, _ = resize_class_and_box_maps(class_pred[0], box_pred[0], size=(args.img_h, args.img_w))
# print('max', tf.reduce_max(box_pred))
# class_map, box_map = class_pred.numpy(), box_pred.numpy()
# del class_pred, box_pred
# 
# coco = CocoGenerator()
# 
# count = 0
# classes = []
# for i in range(box_map.shape[0]):
#     for j in range(box_map.shape[1]):
#         l, t, r, b = box_map[i, j]
#         if np.max([l, t, r, b]) > 1:
#             count += 1
#             if np.argmax(class_map[i, j]) not in classes:
#                 cls_label = np.argmax(class_map[i, j])
#                 cls_name = coco.label_to_name(cls_label)
#                 classes.append(cls_name)
#                 print('class name',cls_name,'\tcoordinates:',l, t, r, b)
#                 img = img.copy()    
#                 cv2.rectangle(img, (round(j - l), round(i - t)), (round(j + r), round(i + b)), (0, 0, 255), 3)
#                 cv2.imshow(f'box{i, j}', img)
# print(count)
# classes = []
# for arg in np.argwhere(class_map > 0.5):
#     cls_label = arg[-1]
#     cls_name = coco.label_to_name(cls_label)
#     if cls_name not in classes:
#         print(cls_name)
#         classes.append(cls_name)
# 
# cv2.waitKey(5000)

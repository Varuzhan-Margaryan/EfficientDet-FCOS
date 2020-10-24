import tensorflow as tf
import argparse
from models.EDet import EDet
from time import time, sleep
import cv2
import numpy as np
from preprocessing import resize_class_and_box_maps
from generators.CocoGenerator import CocoGenerator


data_dir = r'.\data'

# Parser
parser = argparse.ArgumentParser(description="Global parameters for training.")
parser.add_argument("--base_dir", type=str, default="./results", help="directory in which results will be stored")
parser.add_argument("--data_dir", type=str, default=data_dir, help="data directory")
parser.add_argument("--img_h", type=int, default=224, help="size of network input image")
parser.add_argument("--img_w", type=int, default=224, help="size of network input image")
parser.add_argument("--model_name", type=str, default="test", help="name of model")
parser.add_argument("--verbose", type=int, default=1, help="verbosity of model")
args = parser.parse_args()

model = EDet(
    img_size=(args.img_h, args.img_w),
    FPN_depth=3,
    FPN_conv_filters=16,
    clsbox_conv_filters=32,
    data_dir=args.data_dir,
    base_dir=args.base_dir,
    training=False,
    model_name=args.model_name,
    verbose=args.verbose
)
model.ckpt_and_sum_setup()

img = cv2.imread('img.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (args.img_h, args.img_w))

X = tf.expand_dims(img, 0)
class_pred, box_pred = model.forward_pass(X, training=False)
print('max box ', tf.reduce_max(box_pred))
print('max cls ', tf.reduce_max(class_pred))
class_pred, box_pred, _ = resize_class_and_box_maps(class_pred[0], box_pred[0], size=(args.img_h, args.img_w))
print('max', tf.reduce_max(box_pred))
class_map, box_map = class_pred.numpy(), box_pred.numpy()
del class_pred, box_pred

coco = CocoGenerator()

# count = 0
# classes = []
# for i in range(class_map.shape[0]):
#     for j in range(class_map.shape[1]):
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
label_colors = np.random.uniform(0,255,(3,80))
classes = []
for arg in np.argwhere(class_map > 0.5):
    i = arg[0]
    j = arg[1]
    cls_label = arg[-1]
    cls_name = coco.label_to_name(cls_label)
    cls_col = label_colors[cls_label]
    img[i,j] = 0.5*img[i,j]+0.5*cls_col

    if cls_name not in classes:
        print(cls_name)
        classes.append(cls_name)
cv2.imshow(f'imagee', img)
cv2.waitKey(5000)

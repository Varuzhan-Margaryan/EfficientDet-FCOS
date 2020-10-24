from generators.CocoGenerator import CocoGenerator
import cv2
import preprocessing
import numpy as np


#
#
# def show_boxes(img, box_map):
#     for i in range(box_map.shape[0]):
#         for j in range(box_map.shape[1]):
#             l, t, r, b = box_map[i, j]
#             if min(l, t, r, b) > 0:
#                 print(l,t,r,b)
#                 pt1 = (round(2 * (j - l)), round(2 * (i - t)))
#                 pt2 = (round(2 * (j + r)), round(2 * (i + b)))
#                 cv2.rectangle(img, pt1, pt2, (255, 0, 0), 1)
#                 cv2.circle(img, (2 * j, 2 * i), 1, (0, 255, 0))
#
#     cv2.imshow('boxes', img)
#
#
# coco = CocoGenerator(set_name='val2017')
# data_iterator = coco.generate_data()
# for img, class_map, box_map, centerness_map in data_iterator:
#     class_map, box_map, centerness_map = class_map.numpy(), box_map.numpy(), centerness_map.numpy()
#     print('img',img.shape)
#     print('class_map',class_map.shape)
#     print('box_map',box_map.shape)
#     print('centerness_map',centerness_map.shape)
#
#     img = img.astype('float32')
#     cv2.waitKey(1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     cv2.imshow('img', img)
#
#     show_boxes(img, box_map)
#
#     while True:
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break


def show_network_box_map(img, box_map):
    for i in range(box_map.shape[0]):
        for j in range(box_map.shape[1]):
            l, t, r, b = box_map[i, j]
            if min(l, t, r, b) > 0:
                print(l, t, r, b)
                pt1 = (round(2 * (j - l)), round(2 * (i - t)))
                pt2 = (round(2 * (j + r)), round(2 * (i + b)))
                cv2.rectangle(img, pt1, pt2, (255, 0, 0), 1)
                cv2.circle(img, (2 * j, 2 * i), 1, (0, 255, 0))

    cv2.imshow('boxes', img)


def show_box_map(img, box_map, name='box_map'):
    img = img.copy()
    for i in range(0, box_map.shape[0], 1):
        for j in range(0, box_map.shape[1], 1):
            l, t, r, b = box_map[i, j]
            if min(l, t, r, b) > 1:
                pt1 = (round(j - l), round(i - t))
                pt2 = (round(j + r), round(i + b))
                cv2.rectangle(img, pt1, pt2, (255, 0, 0), 1)

    if img.shape[0] < 200:
        img = cv2.resize(img, (3 * img.shape[0], 3 * img.shape[1]))

    cv2.imshow(name, img)


def show_class_map(img, class_map):
    img = img.copy()
    for arg in np.argwhere(class_map > 0.3):
        i, j, label = arg
        cv2.circle(img, (j, i), 1, (label / 80, label / 80, label / 80))

    if img.shape[0] < 200:
        img = cv2.resize(img, (3 * img.shape[0], 3 * img.shape[1]))
    cv2.imshow('class_map', img)


def show_original_boxes(img, boxes):
    for box in boxes:
        x0, y0, x, y = box
        pt1 = (round(x0), round(y0))
        pt2 = (round(x), round(y))
        cv2.rectangle(img, pt1, pt2, (255, 0, 0), 1)

    if img.shape[0] < 200:
        img = cv2.resize(img, (3 * img.shape[0], 3 * img.shape[1]))
    cv2.imshow('boxes', img)


def show_original_masks(img, masks):
    for mask in masks:
        for arg in np.argwhere(mask > 0):
            i, j = arg
            cv2.circle(img, (j, i), 1, (255, 0, 0))

    cv2.imshow('masks', img)


coco = CocoGenerator(set_name='val2017')
data_iterator = coco.generate_data()

for i in coco.image_ids:
    # load
    img, labels, boxes, masks = coco.load_img_and_anns(i)

    if img is None:
        continue

    count = 0
    for x, y, w, h in boxes:
        if w > 1 and h > 1:
            count += 1
    print('len original boxes', len(boxes))
    print('len original big boxes', count)
    img = img.astype('float32')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # show_original_boxes(img, boxes)
    # show_original_masks(img, masks)

    map_size = (112, 112)
    img = cv2.resize(img, map_size)
    class_map, box_map, centerness_map = preprocessing.get_resized_maps(labels, boxes, masks, map_size)
    show_class_map(img,class_map)
    print(class_map.shape)
    print(box_map.shape)
    print(centerness_map.shape)
    print('max bbox', box_map.max())

    show_box_map(img, box_map)

    boxes = preprocessing.get_boxes(class_map, box_map, box_delta=2)
    print('len boxes', len(boxes))

    show_original_boxes(img, boxes)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#
# for i in coco.image_ids:
#     # load
#     img, labels, boxes, masks = coco.load_img_and_anns(i)
#     if img is None:
#         continue
#
#     img = img.astype('float32')
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
#     # show_original_boxes(img, boxes)
#     # show_original_masks(img, masks)
#
#     class_map, box_map, centerness_map = preprocessing.get_class_and_box_maps(labels, boxes, masks)
#     print('box shape', box_map.shape)
#     print('img shape', img.shape)
#
#     # show_box_map(img, box_map)
#     # show_class_map(img,class_map)
#
#     map_size =(112,112)
#     img = cv2.resize(img,map_size)
#     class_map, box_map, centerness_map = preprocessing.resize_class_and_box_maps(class_map, box_map, centerness_map, map_size)
#     class_map, box_map, centerness_map = class_map.numpy(), box_map.numpy(), centerness_map.numpy()
#     print(class_map.shape)
#     print(box_map.shape)
#     print(centerness_map.shape)
#
#
#     show_box_map(img, box_map)
#
#     while True:
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

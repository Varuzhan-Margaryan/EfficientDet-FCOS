import numpy as np
import cv2
import tensorflow as tf

num_classes = 80


#
# def get_class_and_box_maps(labels, boxes, masks):
#     img_shape = masks[0].shape
#     class_map = np.zeros((img_shape[0], img_shape[1], num_classes))
#     box_map = np.zeros((img_shape[0], img_shape[1], 4))
#     centerness_map = np.zeros((img_shape[0], img_shape[1], 1))
#     for i in range(len(labels)):
#         # class map
#         class_map[:, :, labels[i]] += masks[i]
#
#         # box map
#         id = np.indices((img_shape[0], img_shape[1]))
#         x0, y0, w, h = boxes[i]
#         x = x0 + w
#         y = y0 + h
#         l, t, r, b = (id[1] - x0), (id[0] - y0), (x - id[1]), (y - id[0])
#         mask = masks[i].astype('bool')
#         # left
#         box_map[mask, 0] = l[mask]
#         # top
#         box_map[mask, 1] = t[mask]
#         # right
#         box_map[mask, 2] = r[mask]
#         # bottom
#         box_map[mask, 3] = b[mask]
#
#         # centerness map
#         boxed_mask = np.zeros_like(mask, dtype=np.bool)
#         boxed_mask[round(y0 + 1):round(y), round(x0 + 1):round(x)] = True
#         l, t, r, b = l[boxed_mask], t[boxed_mask], r[boxed_mask], b[boxed_mask]
#         centerness_map[boxed_mask, 0] = np.sqrt(
#             (np.minimum(l, r) / (np.maximum(l, r) + 1e-5)) *
#             (np.minimum(t, b) / (np.maximum(t, b) + 1e-5))
#         )
#         centerness_map[np.isnan(centerness_map)] = 0
#
#     return class_map, box_map, centerness_map


# # this called only in testing
# def resize_class_and_box_maps(class_map, box_map, size=(224, 224), threshold=0.3):
#     # resize class map
#     class_map = np.round(cv2.resize(class_map, size))
#
#     # get boxes
#     boxes=[]
#     for arg in np.argwhere(class_map > threshold):
#         i, j, label = arg
#         l, t, r, b = box_map[i, j]
#         x0, y0, x, y = j - t, i - l, j + b, i + r
#         if [x0, y0, x, y] not in boxes:
#             boxes.append([x0, y0, x, y])
#
#
#     # indices and ratios
#     ri = box_map.shape[0] / size[0]
#     rj = box_map.shape[1] / size[1]
#     l = np.expand_dims(box_map[:, :, 0] / rj, axis=-1)
#     r = np.expand_dims(box_map[:, :, 2] / rj, axis=-1)
#     t = np.expand_dims(box_map[:, :, 1] / ri, axis=-1)
#     b = np.expand_dims(box_map[:, :, 3] / ri, axis=-1)
#     return class_map, cv2.resize(np.concatenate([l, t, r, b], axis=-1), size)

# this is called in generator
def get_resized_maps(labels, boxes, masks, size):
    # arrays for maps
    class_map = np.zeros((*size, num_classes))
    box_map = np.zeros((*size, 4))
    centerness_map = np.zeros((*size, 1))

    # ratios
    ri = masks[0].shape[0] / size[0]
    rj = masks[0].shape[1] / size[1]

    # resizing masks
    masks = np.array(masks)
    masks = np.round(tf.image.resize(masks[:, :, :, np.newaxis], size))[:, :, :, 0]

    for i in range(len(labels)):
        # class map
        class_map[:, :, labels[i]] += masks[i]

        # box map

        # get coords of box
        x0, y0, w, h = boxes[i]
        x = x0 + w
        y = y0 + h
        x0, y0, x, y = x0 / rj, y0 / ri, x / rj, y / ri
        # get boxed mask
        boxed_mask = np.zeros_like(masks[i], dtype=np.bool)
        boxed_mask[round(y0 + 1):round(y), round(x0 + 1):round(x)] = True

        # get l,t,r,b
        id = np.indices(size)
        l, t, r, b = (id[1] - x0), (id[0] - y0), (x - id[1]), (y - id[0])

        # left
        box_map[boxed_mask, 0] = l[boxed_mask]
        # top
        box_map[boxed_mask, 1] = t[boxed_mask]
        # right
        box_map[boxed_mask, 2] = r[boxed_mask]
        # bottom
        box_map[boxed_mask, 3] = b[boxed_mask]

        # centerness map
        l, t, r, b = l[boxed_mask], t[boxed_mask], r[boxed_mask], b[boxed_mask]
        centerness_map[boxed_mask, 0] = np.sqrt(
            (np.minimum(l, r) / (np.maximum(l, r) + 1e-5)) *
            (np.minimum(t, b) / (np.maximum(t, b) + 1e-5))
        )
    centerness_map[np.isnan(centerness_map)] = 0
    class_map[class_map > 0.5] = 1
    return class_map, box_map, centerness_map


# this is called to test box_maps
def get_boxes(class_map, box_map, threshold=0.3, box_delta=4):
    boxes = []
    for arg in np.argwhere(class_map > threshold):
        i, j, label = arg
        l, t, r, b = box_map[i, j]
        if min(l, t, r, b) < 1:
            continue

        new_x0, new_y0, new_x, new_y = j - l, i - t, j + r, i + b
        for x0, y0, x, y in boxes:
            if (new_x0 - x0) ** 2 + (new_y0 - y0) ** 2 + (new_x - x) ** 2 + (new_y - y) ** 2 < box_delta:
                break
        else:
            boxes.append([new_x0, new_y0, new_x, new_y])
    return boxes


# this is called to test box_maps
def get_resized_boxes(class_map, box_map, size, threshold=0.3, box_delta=4):
    boxes = get_boxes(class_map, box_map, threshold, box_delta)
    boxes = np.array(boxes)
    if len(boxes) > 0:
        ri = size[0] / class_map.shape[0]
        rj = size[1] / class_map.shape[1]
        boxes[:, 0] *= rj
        boxes[:, 1] *= ri
        boxes[:, 2] *= rj
        boxes[:, 3] *= ri
    return boxes


# this is called to get boxes and classes from network output
def get_boxes_and_classes(class_map, box_map, threshold=0.3, box_delta=4, get_scores=False):
    boxes_with_class = []
    for arg in np.argwhere(class_map > threshold):
        i, j, new_label = arg
        l, t, r, b = box_map[i, j]
        if min(l, t, r, b) < 1:
            continue

        new_x0, new_y0, new_x, new_y = j - l, i - t, j + r, i + b
        for x0, y0, x, y, label in boxes_with_class:
            if new_label == label and \
                    (new_x0 - x0) ** 2 + (new_y0 - y0) ** 2 + (new_x - x) ** 2 + (new_y - y) ** 2 < box_delta:
                break
        else:
            if get_scores:
                boxes_with_class.append([new_x0, new_y0, new_x, new_y, new_label, class_map[i, j, new_label]])
            else:
                boxes_with_class.append([new_x0, new_y0, new_x, new_y, new_label])

    return boxes_with_class


# this is called to get RESIZED boxes and classes from network output
def get_resized_boxes_and_classes(class_map, box_map, size, threshold=0.3, box_delta=4, get_scores=False):
    boxes_with_class = get_boxes_and_classes(class_map, box_map, threshold, box_delta, get_scores)

    if len(boxes_with_class) > 0:
        boxes_with_class = np.array(boxes_with_class)
        ri = size[0] / class_map.shape[0]
        rj = size[1] / class_map.shape[1]
        boxes_with_class[:, 0] *= rj
        boxes_with_class[:, 1] *= ri
        boxes_with_class[:, 2] *= rj
        boxes_with_class[:, 3] *= ri
    return boxes_with_class


# def resize_class_and_box_maps(class_map, box_map, centerness_map, size):
#     # resize class map
#     class_map = tf.image.resize(class_map, size)
#     class_map[class_map >= 0.5] = 1
#     class_map[class_map < 0.5] = 0
#     centerness_map = tf.image.resize(centerness_map, size)
#
#     # create new box map
#     new_box_map = tf.zeros((*size, 4))
#     # indices and ratios
#     ri = box_map.shape[0] / size[0]
#     rj = box_map.shape[1] / size[1]
#     i = np.arange(size[0])
#     j = np.arange(size[1])
#     i1 = np.round(i * ri).astype(np.int32)
#     j1 = np.round(j * rj).astype(np.int32)
#     # fill new box map
#     new_box_map[i, j] = box_map[i1, j1]
#     new_box_map[:, :, [0, 2]] /= ri
#     new_box_map[:, :, [1, 3]] /= rj
#
#     return class_map, new_box_map, centerness_map


def show_predictions(img, boxes_with_class):
    pass


def draw_boxes(image, boxes, colors, classes, write_labels=True):
    for xmin, ymin, xmax, ymax, l in boxes:
        class_name = classes[int(l)]

        xmin, ymin, xmax, ymax = round(xmin), round(ymin), round(xmax), round(ymax)
        color = colors[int(l)]

        if write_labels:
            ret, baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.2, 1)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)

        if write_labels:
            cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
            cv2.putText(image, class_name, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)


def draw_segments(image, class_map, colors, classes):
    masks = class_map > 0.3
    print(masks.shape)
    for l in range(80):
        class_name = classes[int(l)]
        color = colors[int(l)]

        mask = masks[:, :, l]
        image[mask] *= 0.5
        image[mask] += 0.5 * color


def show_segments(image, class_map, colors, classes):
    masks = class_map > 0.5
    print(masks.shape)
    for l in range(80):
        class_name = classes[int(l)]
        color = colors[int(l)]

        mask = masks[:, :, l]
        if mask.sum() > 100:
            im = image[:, :, ::-1].copy()
            im[mask] *= 0.2
            im[mask] += 0.8 #* color
            im = cv2.resize(im, (500, 500))
            cv2.imshow(class_name, im)


def draw_boxes_with_scores(image, boxes, colors, classes):
    for xmin, ymin, xmax, ymax, l, s in boxes:
        class_name = classes[int(l)]

        xmin, ymin, xmax, ymax = round(xmin), round(ymin), round(xmax), round(ymax)
        score = '{:.4f}'.format(s)
        color = colors[int(l)]
        label = '-'.join([class_name, score])

        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.2, 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)


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


def show_box_map(img, box_map, name='box_map', step=1):
    img = img.copy()
    ri = img.shape[0] / box_map.shape[0]
    rj = img.shape[1] / box_map.shape[1]
    for i in range(0, box_map.shape[0], step):
        for j in range(0, box_map.shape[1], step):
            l, t, r, b = box_map[i, j]
            if min(l, t, r, b) > 1:
                pt1 = (round(rj * (j - l)), round(ri * (i - t)))
                pt2 = (round(rj * (j + r)), round(ri * (i + b)))
                cv2.rectangle(img, pt1, pt2, (255, 0, 0), 1)

    if img.shape[0] < 200:
        img = cv2.resize(img, 3 * np.array(img.shape))

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

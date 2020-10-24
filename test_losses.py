import numpy as np
from generators.CocoGenerator import CocoGenerator
import cv2
from preprocessing import *


def show_big(name, img):
    img = cv2.resize(img, (500, 500))
    cv2.imshow(name, img)


def loss(class_pred, centerness_pred, box_pred, class_target, box_target, centerness_target):
    # class_pred =   [0.1, 0.001, 0.89, 0.99, 0]
    # class_target = [0,   0,     1,    1,    0]
    # box_pred = [13, 24 ,90, 12] (l,t,r,b) (box_traget-same)

    # print('\n\nsizes:, clas_pred',class_pred.shape, 'centerness_pred',centerness_pred.shape,
    #       '\nbox_pred', box_pred.shape, 'class_target',class_target.shape,
    #       '\nbox_target', box_target.shape, 'centerness_target',centerness_target.shape,
    #       )
    # print('\nminmax:, clas_pred',class_pred.numpy().min(),class_pred.numpy().max(),
    #       '\ncenterness_pred',centerness_pred.numpy().min(),centerness_pred.numpy().max(),
    #       '\nbox_pred', box_pred.numpy().min(), box_pred.numpy().max(),
    #       '\nclass_target',class_target.numpy().min(),class_target.numpy().max(),
    #       '\nbox_target', box_target.numpy().min(), box_target.numpy().max(),
    #       '\ncenterness_target',centerness_target.numpy().min(),centerness_target.numpy().max()
    #       )

    # box coordinates
    # class_target, box_target = resize_class_and_box_maps_tf(class_target, box_target, tf.shape(box_pred)[1:3])

    l, t, r, b = box_target[:, :, :, 0], box_target[:, :, :, 1], box_target[:, :, :, 2], box_target[:, :, :, 3]
    lp, tp, rp, bp = box_pred[:, :, :, 0], box_pred[:, :, :, 1], box_pred[:, :, :, 2], box_pred[:, :, :, 3]
    del box_pred, box_target

    # not empty pixels
    not_empty_mask = tf.greater(tf.reduce_sum(class_target, axis=-1), 0.5)
    num_pos = tf.reduce_sum(tf.cast(not_empty_mask, tf.float32))

    # loss for class_pred
    focal_loss = -tf.reduce_sum(
        class_target * (1 - class_pred) ** 2 * tf.math.log(class_pred) +
        (1 - class_target) * class_pred ** 2 * tf.math.log(1 - class_pred)) / num_pos

    # loss for box_pred
    intersection = (tf.minimum(t, tp) + tf.minimum(b, bp)) * (tf.minimum(l, lp) + tf.minimum(r, rp))
    area_traget = (t + b) * (r + l)
    area_pred = (tp + bp) * (rp + lp)
    union = tf.maximum(1e-10, area_pred + area_traget - intersection)
    iou_loss = 1 - tf.reduce_mean(intersection[not_empty_mask] / union[not_empty_mask])

    centerness_loss = -tf.reduce_mean(centerness_target * tf.math.log(centerness_pred) +
                                      (1 - centerness_target) * tf.math.log(1 - centerness_pred))

    import numpy as np
    if not (np.all(0 < focal_loss.numpy() < 1e10) and np.all(0 < iou_loss.numpy() < 1e10) and np.all(
            0 < centerness_loss.numpy() < 1e10)):
        print('ERRRRRRRRRRRRRRROR')
        print('\n\nsizes: \nfocal_loss', focal_loss.shape, ',iou_loss', iou_loss.shape,
              ',centerness_loss', centerness_loss.shape)
        print('\nminmax:\nfocal_loss', focal_loss.numpy().min(), focal_loss.numpy().max(),
              '\niou_loss', iou_loss.numpy().min(), iou_loss.numpy().max(),
              '\ncenterness_loss', centerness_loss.numpy().min(), centerness_loss.numpy().max()
              )
        assert 1 == 0

    return focal_loss, iou_loss, centerness_loss  # can add coefficients


coco = CocoGenerator(data_dir='.\data', set_name='val2017')
colors = np.random.uniform(0, 1, size=(80, 3))  # random colors for classes
classes = coco.labels

for img, class_map, box_map, centerness_map in coco.generate_data():
    print(
        f'Shapes: \n img:{img.shape} \n class_map:{class_map.shape} \n box_map:{box_map.shape} \n centerness_map:{centerness_map.shape}')
    class_target = np.expand_dims(class_map, 0).astype('float32')
    class_pred = class_target.copy()
    class_pred[class_pred == 1] = 0.9999
    class_pred[class_pred == 0] = 0.0001

    centerness_target = np.expand_dims(centerness_map, 0).astype('float32')
    centerness_pred = centerness_target.copy()
    centerness_pred[centerness_pred == 1] = 0.9
    centerness_pred[centerness_pred == 0] = 0.01

    box_target = np.expand_dims(box_map, 0).astype('float32')
    box_pred = box_target.copy()
    focal_loss, iou_loss, centerness_loss = loss(class_pred, centerness_pred, box_pred, class_target, box_target,
                                                 centerness_target)
    print('LOSS:', focal_loss.numpy(), iou_loss.numpy(), centerness_loss.numpy())

    class_pred = class_target.copy()
    class_pred[class_pred == 1] = 0.8
    class_pred[class_pred == 0] = 0.2

    centerness_pred = centerness_target.copy()
    centerness_pred[centerness_pred == 1] = 0.8
    centerness_pred[centerness_pred == 0] = 0.2

    box_pred = box_target.copy()+3
    focal_loss, iou_loss, centerness_loss = loss(class_pred, centerness_pred, box_pred, class_target, box_target,
                                                 centerness_target)
    print('LOSS:', focal_loss.numpy(), iou_loss.numpy(), centerness_loss.numpy())

    box_pred = np.random.uniform(0.0001, 0.9999, box_target.shape).astype('float32')
    class_pred = np.random.uniform(0.0001, 0.9999, class_target.shape).astype('float32')
    centerness_pred = np.random.uniform(0.0001, 0.9999, centerness_target.shape).astype('float32')

    focal_loss, iou_loss, centerness_loss = loss(class_pred, centerness_pred, box_pred, class_target, box_target,
                                                 centerness_target)
    print('LOSS:', focal_loss.numpy(), iou_loss.numpy(), centerness_loss.numpy())

    # show image
    # show_big('img', img[:, :, ::-1])

    # show class_map
    # for i in range(80):
    #     cls = class_map[:, :, i]
    #     if cls.max() > 0:
    #         show_big(f'class_map{i}', cls)

    boxes_classes = get_resized_boxes_and_classes(class_map, box_map, img.shape[:2])

    img1 = img.copy()
    draw_boxes(img1, boxes_classes, colors, classes)
    show_big('img1', img1[:, :, ::-1])

    # boxes = boxes_classes[:, :-1]
    # boxes[:,2] -= boxes[:,0]
    # boxes[:,3] -= boxes[:,1]
    # labels = boxes_classes[:, -1].astype('int')
    # class_map1, box_map1, centerness_map1 = get_resized_maps(labels, boxes, class_map[:, :, labels], box_map.shape[:2])
    # print(np.mean(class_map1 - class_map), np.mean(box_map1 - box_map), np.mean(centerness_map1 - centerness_map))
    # assert np.all(class_map1==class_map) and np.all(box_map1==box_map) and np.all(centerness_map1==centerness_map)

    show_box_map(img[:, :, ::-1], box_map)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

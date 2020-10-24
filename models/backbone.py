# import tensorflow as tf
# from models.efficientnet import EfficientNetB0
#
# img_size = (224, 224)
#
#
# def create_backbone(img_size):
#     inp = tf.keras.layers.Input(shape=(*img_size, 3))
#     out = EfficientNetB0(input_tensor=inp, input_shape=(*img_size, 3), layers=tf.keras.layers,
#                          backend=tf.keras.backend)
#     return tf.keras.models.Model(inputs=[inp], outputs=out, name='efficientnet-B0')
#
#
# _backbone = create_backbone(img_size)
#
#
# @tf.function
# def backbone(x):
#     return _backbone(x)

def build_backbone():
    from tensorflow.keras.applications import EfficientNetB0
    import tensorflow as tf

    model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    red1 = model.get_layer('block1a_activation').output
    red2 = model.get_layer('block2b_activation').output
    red3 = model.get_layer('block3b_activation').output
    red4 = model.get_layer('block5c_activation').output
    red5 = model.get_layer('top_activation').output

    return tf.keras.Model(inputs=model.inputs, outputs=[red1, red2, red3, red4, red5], trainable=False)



# # test
# import cv2
# from time import time
# img = tf.convert_to_tensor(cv2.resize(cv2.imread(r'..\cat.jpg') / 255.0, (256, 256)).reshape((1, 256, 256, 3)))
#
# for i in range(10):
#     t0 = time()
#     out = backbone(img)
#     print(time() - t0)
#
# for o in out:
#     print('back model', o.numpy().mean())
# #
# # # model.summary()
# # for l in model.layers:
# #     # print(l.name)
# #     print(l.name,'\t', l.output_shape)

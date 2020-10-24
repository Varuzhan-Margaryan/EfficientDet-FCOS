from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf
import cv2

model = EfficientNetB0(include_top=False, weights='imagenet',input_shape=(224,224,3))


img = tf.convert_to_tensor(cv2.resize(cv2.imread('cat.jpg') / 255.0, (256, 256)).reshape((1, 256, 256, 3)))

out = model(img)
print(out.numpy().mean())

# model.summary()
for l in model.layers:
    # print(l.name)
    print(l.name,'\t', l.output_shape)
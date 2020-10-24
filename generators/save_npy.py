import tensorflow as tf
import os
from generators.CocoGenerator import CocoGenerator
from time import time
from tqdm import tqdm
import numpy as np

data_dir = '/home/varuzh/my/coco/2017'
target_dir = '/home/varuzh/my/edet'
img_size = (224, 224)
map_size = (112, 112)
batch_size = 10
val_batch_size = 10

train_generator = CocoGenerator(data_dir, 'val2017', img_size, shuffle=True)
val_generator = CocoGenerator(data_dir, 'val2017', img_size, shuffle=False)

num_classes = train_generator.num_classes()
out_sizes = ((*img_size, 3), (*map_size, num_classes), (*map_size, 4), (*map_size, 1))

training_dataset = tf.data.Dataset.from_generator(train_generator.generate_data,
                                                  (tf.float32, tf.float32, tf.float32, tf.float32),
                                                  out_sizes)

val_dataset = tf.data.Dataset.from_generator(val_generator.generate_data,
                                             (tf.float32, tf.float32, tf.float32, tf.float32), out_sizes)

training_dataset = training_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(val_batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)

# Write the `tf.train.Example` observations to the file.
with tf.io.TFRecordWriter(os.path.join(target_dir,'train')) as writer:
    for img, class_target, box_target, centerness_target in tqdm(training_dataset):
        print('start')
        writer.write(img)

        print('---------------------')


#
# for img, class_target, box_target, centerness_target in tqdm(training_dataset):
#     print('start')
#     print(centerness_target)
#     batch_x = np.array(img)
#     batch_y = np.array([class_target, box_target, centerness_target])
#     print(np.array(batch_x).shape)
#     print(np.array(batch_y).shape)
#     print('---------------------')

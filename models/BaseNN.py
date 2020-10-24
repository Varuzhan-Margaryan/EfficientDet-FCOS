import tensorflow as tf
from abc import abstractmethod
import os
from generators.CocoGenerator import CocoGenerator
from time import time


class BaseNN(tf.keras.Model):
    def __init__(self, map_size=(112, 112), img_size=(244, 244), data_dir=r'data', epochs=1,
                 learning_rate=1e-4,
                 batch_size=32, val_batch_size=64, base_dir='results', training=True,
                 max_to_keep=5, model_name='test', start_step=0, verbose=1, *args, **kwargs):
        super(BaseNN, self).__init__(*args, **kwargs)
        # init variables
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.max_to_keep = max_to_keep
        self.verbose = verbose
        self.data_dir = data_dir
        self.img_size = img_size
        self.map_size = map_size
        self.training = training

        # make necessary directories
        self.make_dirs(base_dir, model_name)

        # make global variable for training step
        self.step = start_step
        self.step_tf = tf.Variable(start_step, trainable=False, name="step", dtype=tf.int64)

    def make_dirs(self, base_dir, model_name):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        model_dir = os.path.join(base_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.save_dir = os.path.join(model_dir, 'saved_models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.checkpoint_dir = os.path.join(model_dir, "checkpoints")
        self.checkpoint_save_path = os.path.join(self.checkpoint_dir, "my_model")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.summary_dir = os.path.join(model_dir, "summaries")
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

    def get_data_iterators(self):
        self.train_generator = CocoGenerator(self.data_dir, 'train2017', self.img_size, shuffle=True)
        self.val_generator = CocoGenerator(self.data_dir, 'val2017', self.img_size, shuffle=False)
        self.test_generator = CocoGenerator(self.data_dir, 'test2017', self.img_size, shuffle=False)

        self.num_classes = self.train_generator.num_classes()
        out_sizes = ((*self.img_size, 3), (*self.map_size, self.num_classes), (*self.map_size, 4), (*self.map_size, 1))

        self.training_dataset = tf.data.Dataset.from_generator(self.train_generator.generate_data,
                                                               (tf.float32, tf.float32, tf.float32, tf.float32),
                                                               out_sizes)

        self.val_dataset = tf.data.Dataset.from_generator(self.val_generator.generate_data,
                                                          (tf.float32, tf.float32, tf.float32, tf.float32), out_sizes)

        self.test_dataset = tf.data.Dataset.from_generator(self.train_generator.generate_test_data,
                                                           (tf.float32), (*self.img_size, 3))

        self.training_dataset = self.training_dataset.batch(self.batch_size, drop_remainder=True).repeat(
            self.epochs).prefetch(tf.data.experimental.AUTOTUNE)
        self.val_dataset = self.val_dataset.batch(self.val_batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
        self.test_dataset = self.test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def ckpt_and_sum_setup(self):
        # directories
        train_summary_dir = os.path.join(self.summary_dir, "train")
        val_summary_dir = os.path.join(self.summary_dir, "validation")
        # writers
        self.train_writer = tf.summary.create_file_writer(train_summary_dir)
        self.val_writer = tf.summary.create_file_writer(val_summary_dir)

        self.optimizer = tf.optimizers.Adam(self.learning_rate)

        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        latest = tf.train.latest_checkpoint(self.checkpoint_prefix)

        # Load the previously saved weights
        root = tf.train.Checkpoint(optimizer=self.optimizer, trainable_variables=self.trainable_variables,
                                   step=self.step_tf)
        self.step = self.step_tf.numpy()
        print('step',self.step)
        self.manager = tf.train.CheckpointManager(root, directory=self.checkpoint_prefix, max_to_keep=self.max_to_keep)
        if latest is not None:
            print("[*] Restoring model...")
            root.restore(latest)

    def train_model(self, display_step, validation_step, checkpoint_step, summary_step):
        @tf.function
        def train_step(img, class_target, box_target, centerness_target):
            with tf.GradientTape() as tape:
                class_pred, centerness_pred, box_pred = self.forward_pass(img, self.training)
                class_loss, box_loss, centerness_loss = self.loss(class_pred, centerness_pred, box_pred, class_target,
                                                                  box_target, centerness_target)
                total_loss = class_loss + box_loss + centerness_loss

            gradients = tape.gradient(total_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return class_loss, box_loss, centerness_loss, total_loss

        ################################################################################################################

        if self.verbose > 2:
            print('I starting training')
            count = 0
            for var in self.trainable_variables:
                if self.verbose > 5:
                    print(var.name, var.numpy().shape)
                count += var.numpy().size
            print('I Trainable variables:', count)

        # TRAIN LOOP
        for img, class_target, box_target, centerness_target in self.training_dataset:

            class_loss, box_loss, centerness_loss, total_loss = train_step(img, class_target, box_target,
                                                                           centerness_target)

            if self.step % display_step == 0 and self.verbose > 0:
                self.display_summary(box_loss, centerness_loss, class_loss, total_loss)

            if self.step % summary_step == 0:
                self.write_train_summary(box_loss, centerness_loss, class_loss, total_loss)

            if self.step % validation_step == 0:
                self.write_display_val_summary()

            if self.step % checkpoint_step == 0 and self.step != 0:
                self.save_checkpoint()

            self.step += 1

    def save_checkpoint(self):
        self.step_tf.assign(self.step)
        self.manager.save()
        if self.verbose > 0:
            print('I Saved checkpoint')

    def write_display_val_summary(self):
        val_batches = 5
        class_loss_val, box_loss_val, centerness_loss_val = 0, 0, 0
        for img_val, class_target_val, box_target_val, center_target_val in self.val_dataset.take(val_batches):
            class_pred_val, centerness_pred_val, box_pred_val = self.forward_pass(img_val)
            class_l, box_l, center_l = self.loss(class_pred_val, centerness_pred_val, box_pred_val,
                                                 class_target_val, box_target_val, center_target_val)
            # class_l, box_l, center_l = class_l.numpy(), box_l.numpy(), center_l.numpy()
            class_loss_val += class_l
            box_loss_val += box_l
            centerness_loss_val += center_l
        class_loss_val /= val_batches
        box_loss_val /= val_batches
        centerness_loss_val /= val_batches
        total_loss_val = class_loss_val + box_loss_val + centerness_loss_val

        with self.val_writer.as_default():
            tf.summary.scalar('class_loss', class_loss_val, step=self.step)
            tf.summary.scalar('box_loss', box_loss_val, step=self.step)
            tf.summary.scalar('centerness_loss', centerness_loss_val, step=self.step)
            tf.summary.scalar('total_loss', total_loss_val, step=self.step)
        if self.verbose > 0:
            print(
                f"Step: {self.step}\tVal Loss: {class_loss_val.numpy(), box_loss_val.numpy(), centerness_loss_val.numpy(), total_loss_val.numpy()}")

    def display_summary(self, box_loss, centerness_loss, class_loss, total_loss):
        print(
            f"\nStep: {self.step}\tLoss: {class_loss.numpy(), box_loss.numpy(), centerness_loss.numpy(), total_loss.numpy()}")

    def write_train_summary(self, box_loss, centerness_loss, class_loss, total_loss):
        with self.train_writer.as_default():
            tf.summary.scalar('class_loss', class_loss, step=self.step)
            tf.summary.scalar('box_loss', box_loss, step=self.step)
            tf.summary.scalar('centerness_loss', centerness_loss, step=self.step)
            tf.summary.scalar('total_loss', total_loss, step=self.step)

    def call(self, x, training=None, **kwargs):
        return self.forward_pass(x, training=training)

    # def test_model(self, count=200):
    #     from preprocessing import load_video_async, preprocess_frame_for_test, get_absolute_coords, draw_bbox
    #     import cv2
    #     videos = self.test_data_dir
    #     breaked = False
    #     for dir in os.listdir(videos):
    #         try:
    #             dir = os.path.join(videos, dir)
    #             frames, bboxes = load_video_async(dir, count=count)
    #
    #             x_test, resx, resy, ratx, raty = preprocess_frame_for_test(
    #                 frames[0], frames[1], bboxes[0])
    #             # del bboxes
    #             for i in tqdm(range(len(frames) - 1)):
    #                 x = x_test[np.newaxis, :].astype('float16') / 255
    #                 pred = self.model(x)[0].numpy().astype('float')
    #                 pred = get_absolute_coords(pred, resx, resy, ratx, raty)
    #                 # print('absolute prediction =', pred)
    #
    #                 img = draw_bbox(frames[i], pred)
    #                 cv2.imshow(dir[-3:], img)
    #                 key = (cv2.waitKey(1) & 0xFF)
    #                 if key == ord('q'):
    #                     breaked = True
    #                     break
    #                 elif key == ord("n"):
    #                     break
    #                 elif key == ord("t"):
    #                     pred = bboxes[i]
    #
    #                 x_test, resx, resy, ratx, raty = preprocess_frame_for_test(
    #                     frames[i], frames[i + 1], pred)
    #
    #             cv2.destroyAllWindows()
    #             if breaked:
    #                 break
    #         except:
    #             pass
    #     cv2.destroyAllWindows()

    @abstractmethod
    @tf.function
    def forward_pass(self, x, training=True):
        pass

    @abstractmethod
    @tf.function
    def loss(self, class_pred, centerness_pred, box_pred, class_target, box_target, centerness_target):
        pass

import tensorflow as tf


class BiFPN(tf.keras.layers.Layer):
    def __init__(self,repeats=3, num_inputs=5, conv_filters=64, conv_kernel_size=3):
        super(BiFPN, self).__init__()
        self.num_inputs = num_inputs
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.repeats = repeats

        ######################## build #####################

        # top-down weights
        self.td_ws = []
        for i in range(1, self.num_inputs - 1):
            self.td_ws.append([self.add_weight(f"w_td_{i}_1", shape=(), initializer='random_normal', trainable=True),
                               self.add_weight(f"w_td_{i}_2", shape=(), initializer='random_normal', trainable=True)])

        # bottom-up weights
        self.bu_ws = []

        ## first node
        self.bu_ws.append([self.add_weight("w_bu_0_1", shape=(), initializer='random_normal', trainable=True),
                           self.add_weight("w_bu_0_2", shape=(), initializer='random_normal', trainable=True)])

        for i in range(1, self.num_inputs - 1):
            self.bu_ws.append([self.add_weight(f"w_bu_{i}_1", shape=(), initializer='random_normal', trainable=True),
                               self.add_weight(f"w_bu_{i}_2", shape=(), initializer='random_normal', trainable=True),
                               self.add_weight(f"w_bu_{i}_3", shape=(), initializer='random_normal', trainable=True)])

        ## last node
        self.bu_ws.append(
            [self.add_weight(f"w_bu_{self.num_inputs - 1}_1", shape=(), initializer='random_normal', trainable=True),
             self.add_weight(f"w_bu_{self.num_inputs - 1}_2", shape=(), initializer='random_normal', trainable=True)])

        # convolutions
        self.td_convs = [
            tf.keras.layers.SeparableConv2D(self.conv_filters, self.conv_kernel_size, padding='same')
            for _ in range(self.num_inputs - 2)]

        self.bu_convs = [tf.keras.layers.SeparableConv2D(self.conv_filters, self.conv_kernel_size, padding='same')
                         for _ in range(self.num_inputs)]
        self.batchnorms = [tf.keras.layers.BatchNormalization() for _ in range(8)]

    def fuse_features_2(self,in1, in2, w1, w2, conv, bn):
        return tf.nn.relu(bn(conv(
            (w1 * in1 + w2 * tf.image.resize(in2, tf.shape(in1)[1:3])) /
            (w1 + w2 + 1e-5)
        )))

    def fuse_features_3(self,in1, in2, in3, w1, w2, w3, conv, bn):
        return tf.nn.relu(bn(conv(
            (w1 * in1 + w2 * in2 + w3 * tf.image.resize(in3, tf.shape(in1)[1:3])) /
            (w1 + w2 + w3 + 1e-5)
        )))
    def call(self, input, **kwargs):
        for _ in range(self.repeats):
            # top-down part
            ws = self.td_ws[0]
            td_0 = self.fuse_features_2(input[1], input[0], ws[1], ws[0], self.td_convs[0], self.batchnorms[0])

            ws = self.td_ws[1]
            td_1 = self.fuse_features_2(input[2], td_0, ws[1], ws[0], self.td_convs[1], self.batchnorms[1])

            ws = self.td_ws[2]
            td_2 = self.fuse_features_2(input[3], td_1, ws[1], ws[0], self.td_convs[2], self.batchnorms[2])
            ###############################################################################################################
            # bottom-up part
            ws = self.bu_ws[4]
            input[4] = self.fuse_features_2(input[4], td_2, ws[0], ws[1], self.bu_convs[4], self.batchnorms[3])

            ws = self.bu_ws[3]
            input[3] = self.fuse_features_3(td_2, input[3], input[4], ws[0], ws[1], ws[2], self.bu_convs[3], self.batchnorms[4])

            ws = self.bu_ws[2]
            input[2] = self.fuse_features_3(td_1, input[2], input[3], ws[0], ws[1], ws[2], self.bu_convs[2], self.batchnorms[5])

            ws = self.bu_ws[1]
            input[1] = self.fuse_features_3(td_0, input[1], input[2], ws[0], ws[1], ws[2], self.bu_convs[1], self.batchnorms[6])

            ws = self.bu_ws[0]
            input[0] = self.fuse_features_2(input[0], input[1], ws[0], ws[1], self.bu_convs[0], self.batchnorms[7])

        return input

    # def call(self, input, **kwargs):
    #     # top-down part
    #     td_array = []
    #
    #     ## first td node
    #     td_array.append(self.fuse_features_2(input[1], input[0], self.td_ws[0][1], self.td_ws[0][0], self.td_convs[0]))
    #
    #     for i in range(1, self.num_inputs - 2):
    #         ws = self.td_ws[i]
    #         td_array.append(self.fuse_features_2(input[i + 1], td_array[-1], ws[1], ws[0], self.td_convs[i]))
    #
    #     ###############################################################################################################
    #     # bottom-up part
    #     bu_array = []
    #
    #     ## last bu node
    #     ws = self.bu_ws[-1]
    #     bu_array.append(self.fuse_features_2(input[-1], td_array[-1], ws[0], ws[1], self.bu_convs[-1]))
    #
    #     for i in range(self.num_inputs - 2, 0, -1):
    #         ws = self.bu_ws[i]
    #         bu_array.append(
    #             self.fuse_features_3(td_array[i - 1], input[i], bu_array[-1], ws[0], ws[1], ws[2],
    #                                  self.bu_convs[i]))
    #
    #     ## first bu node
    #     ws = self.bu_ws[0]
    #     bu_array.append(self.fuse_features_2(input[0], bu_array[1], ws[0], ws[1], self.bu_convs[0]))
    #
    #     return bu_array[::-1]

    # def call(self, input, **kwargs):
    #     # input resizing
    #     max_shape = tf.shape(input[-1])[0]
    #     in_array = tf.TensorArray(tf.float32, size=self.num_inputs, dynamic_size=False, clear_after_read=False,
    #                               name='in_array')
    #     for i in range(self.num_inputs):
    #         shape = max_shape / 2 ** (self.num_inputs - 1 - i)
    #         start = (max_shape - shape) // 2
    #         in_array.write(i,
    #                        input[i][start:start + shape, start:start + shape])
    #
    #     ###############################################################################################################
    #     # top-down part
    #     td_array = tf.TensorArray(tf.float32, size=self.num_inputs - 2, dynamic_size=False, clear_after_read=False,
    #                               name='td_array')
    #
    #     ## first td node
    #     td_array.write(0, self.fuse_features_2(in_array.read(1), in_array.read(0), *self.td_ws[0][::-1]))
    #     for i in range(1, self.num_inputs - 1):
    #         td_array.write(i, self.fuse_features_2(td_array.read(i - 1), in_array.read(i + 1), *self.td_ws[i][::-1]))
    #
    #     ###############################################################################################################
    #     # bottom-up part
    #     bu_array = tf.TensorArray(tf.float32, size=self.num_inputs, dynamic_size=False, clear_after_read=False,
    #                               name='bu_array')
    #
    #     ## last bu node
    #     bu_array.write(self.num_inputs - 1,
    #                    self.fuse_features_2(in_array.read(self.num_inputs - 1), td_array.read(self.num_inputs - 3),
    #                                         *self.bu_ws[-1]))
    #     for i in range(self.num_inputs - 2, 0, -1):
    #         bu_array.write(i, self.fuse_features_3(td_array.read(i - 1), in_array.read(i), bu_array.read(i + 1),
    #                                                *self.bu_ws[i]))
    #     ## first bu node
    #     bu_array.write(0, self.fuse_features_2(in_array.read(0), bu_array.read(1), *self.bu_ws[0]))
    #
    #     ###############################################################################################################
    #     # output resizing
    #     out = tf.Variable(trainable=False, shape=(self.num_inputs, max_shape, max_shape, self.conv_filters))
    #     for i in range(self.num_inputs):
    #         shape = max_shape / 2 ** (self.num_inputs - 1 - i)
    #         pad = (max_shape - shape) // 2
    #         out[i] = tf.pad(bu_array.read(i), [[pad, pad], [pad, pad]])
    #
    #     return out

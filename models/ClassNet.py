import tensorflow as tf


class ClassNet(tf.keras.layers.Layer):
    def __init__(self, class_count=80, conv_filters=64, convs_num=4, conv_kernel_size=3):
        super(ClassNet, self).__init__()
        # init variables
        self.class_cunt = class_count
        self.convs_num = convs_num
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size

        # create layers
        self.convs = [tf.keras.layers.Conv2D(self.conv_filters, self.conv_kernel_size, padding='same')
                      for _ in range(self.convs_num)]
        self.batchnorms = [tf.keras.layers.BatchNormalization() for _ in range(self.convs_num)]
        self.class_conv = tf.keras.layers.Conv2D(self.class_cunt, self.conv_kernel_size, padding='same')
        self.centerness_conv = tf.keras.layers.Conv2D(1, self.conv_kernel_size, padding='same')

        #
        # for conv in self.convs+[self.class_conv,self.centerness_conv,self.batchnorm]:
        #     self.trainable_variables.append(conv.trainable_variables)

    #
    # def set_weights(self,weights):
    #     i=0
    #     for conv in self.convs+[self.class_conv,self.centerness_conv,self.batchnorm]:
    #         l = len(conv.trainable_variables)
    #         conv.set_weights(weights[i:i+l])
    #         i+=l

    def call(self, input, **kwargs):
        for bn, conv in zip(self.batchnorms, self.convs):
            input = tf.nn.relu(bn(conv(input)))

        # class_pred = tf.clip_by_value(tf.nn.sigmoid(self.class_conv(input)), 1e-5, 1 - 1e-5)
        # centerness_pred = tf.clip_by_value(tf.nn.sigmoid(self.centerness_conv(input)), 1e-5, 1 - 1e-5)
        class_pred = tf.nn.sigmoid(self.class_conv(input))
        centerness_pred = tf.nn.sigmoid(self.centerness_conv(input))

        return class_pred, centerness_pred

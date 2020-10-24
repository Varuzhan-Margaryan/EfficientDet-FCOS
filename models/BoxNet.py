import tensorflow as tf


class BoxNet(tf.keras.layers.Layer):
    def __init__(self, conv_filters=128, convs_num=4, conv_kernel_size=3):
        super(BoxNet, self).__init__()
        # init variables
        self.convs_num = convs_num
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size

        # create layers
        self.convs = [tf.keras.layers.Conv2D(self.conv_filters, self.conv_kernel_size, padding='same')
                      for _ in range(self.convs_num)]
        self.batchnorms = [tf.keras.layers.BatchNormalization() for _ in range(self.convs_num)]
        self.box_conv = tf.keras.layers.Conv2D(4, self.conv_kernel_size, padding='same')

    def call(self, input, **kwargs):
        for bn, conv in zip(self.batchnorms, self.convs):
            input = tf.nn.relu(bn(conv(input)))

        box_pred = tf.exp(self.box_conv(input))
        return box_pred

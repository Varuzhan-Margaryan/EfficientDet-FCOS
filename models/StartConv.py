import tensorflow as tf


class StartConv(tf.keras.layers.Layer):
    def __init__(self, num_inputs=5, conv_filters=64, conv_kernel_size=3):
        super(StartConv, self).__init__()
        self.num_inputs = num_inputs
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size

        ######################## build #####################
        # convolutions
        self.drops = [tf.keras.layers.Dropout(0.5)
                         for _ in range(self.num_inputs)]
        self.convs = [tf.keras.layers.Conv2D(self.conv_filters, self.conv_kernel_size, padding='same')
                         for _ in range(self.num_inputs)]
        self.batchnorms = [tf.keras.layers.BatchNormalization() for _ in range(self.num_inputs)]
        self.relu = tf.nn.relu

    def call(self, inputs, training=True, **kwargs):
        for i in range(len(inputs)):
            inputs[i] = self.relu(self.batchnorms[i](self.convs[i](self.drops[i](inputs[i], training))))
        return inputs
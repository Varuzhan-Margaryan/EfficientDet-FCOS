import tensorflow as tf
from .BaseNN import BaseNN
from .StartConv import StartConv
from .backbone import build_backbone
from .BiFPN import BiFPN
from .ClassNet import ClassNet
from .BoxNet import BoxNet


class EDet(BaseNN):

    def __init__(self, map_size=(112, 112), backbone_model='b0', img_size=(512, 512), FPN_depth=3, FPN_inputs=5,
                 FPN_conv_filters=16, clsbox_conv_filters=32, classnet_convs_num=4, boxnet_convs_num=4,
                 data_dir=r'data', epochs=1, learning_rate=1e-4,
                 batch_size=8, val_batch_size=8, base_dir='results', training=True, max_to_keep=5, model_name='test',
                 start_step=0, verbose=1, *args, **kwargs):
        super(EDet, self).__init__(map_size, img_size, data_dir, epochs, learning_rate, batch_size, val_batch_size,
                                   base_dir, training, max_to_keep, model_name, start_step, verbose, *args, **kwargs)

        self.FPN_depth = FPN_depth
        self.FPN_inputs = FPN_inputs
        self.FPN_conv_filters = FPN_conv_filters
        self.clsbox_conv_filters = clsbox_conv_filters
        self.classnet_convs_num = classnet_convs_num
        self.boxnet_convs_num = boxnet_convs_num
        self.backbone_model = backbone_model  # name

        # Creating layers
        self.create_layers()

        # test pass
        self.call(tf.ones((1, *img_size, 3)))

    def create_layers(self):
        self.backbone = build_backbone()
        self.start_conv = StartConv(num_inputs=self.FPN_inputs, conv_filters=self.FPN_conv_filters, conv_kernel_size=3)
        self.BiFPN = BiFPN(repeats=self.FPN_depth, num_inputs=self.FPN_inputs, conv_filters=self.FPN_conv_filters,
                           conv_kernel_size=3)
        self.classnet = ClassNet(class_count=80, conv_filters=self.clsbox_conv_filters,
                                 convs_num=self.classnet_convs_num, conv_kernel_size=3)
        self.boxnet = BoxNet(conv_filters=self.clsbox_conv_filters, convs_num=self.boxnet_convs_num, conv_kernel_size=3)
        self.feture_maps_size = (112, 112)

    @tf.function
    def forward_pass(self, X, training=True):
        # backbone CNN
        X = self.backbone(X)  # getting image returning 5 feature maps

        # start convolutions
        X = self.start_conv(X,training)

        # feature pyramid networks
        X = self.BiFPN(X)

        # resize feature maps
        for i in range(0, self.FPN_inputs):
            X[i] = tf.image.resize(X[i], self.feture_maps_size)  # todo: here can be conv layers in place of resize

        # sum feats
        X = X[0] + X[1] + X[2] + X[3] + X[4]  # FPN_out

        ############################################################################################################
        # classnet and boxnet
        class_pred, centerness_pred = self.classnet(X)
        box_pred = self.boxnet(X)

        if training:
            return class_pred, centerness_pred, box_pred
        else:
            return class_pred, centerness_pred * box_pred

    @tf.function
    def loss(self, class_pred, centerness_pred, box_pred, class_target, box_target, centerness_target):
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


        # mse
        # class_target, box_target, centerness_target = tf.ones_like(class_pred), tf.ones_like(box_pred), tf.ones_like(centerness_pred)
        #
        # return tf.reduce_mean((class_pred-class_target)**2), tf.reduce_mean((box_pred-box_target)**2), tf.reduce_mean((centerness_pred-centerness_target)**2)



        # class_target, box_target, centerness_target = tf.zeros_like(class_pred), tf.zeros_like(box_pred), tf.zeros_like(centerness_pred)
        l, t, r, b = box_target[:, :, :, 0], box_target[:, :, :, 1], box_target[:, :, :, 2], box_target[:, :, :, 3]
        lp, tp, rp, bp = box_pred[:, :, :, 0], box_pred[:, :, :, 1], box_pred[:, :, :, 2], box_pred[:, :, :, 3]
        del box_pred, box_target

        # not empty pixels
        not_empty_mask = tf.greater(tf.reduce_sum(class_target, axis=-1), 0.5)
        num_pos = tf.reduce_sum(tf.cast(not_empty_mask, tf.float32))

        # loss for class_pred
        term1 = -(1 - class_pred) ** 2 * tf.math.log(class_pred)  # min 1
        term2 = -class_pred ** 2 * tf.math.log(1 - class_pred)  # min 0
        focal_loss = tf.reduce_sum(
            class_target * term1 + (1 - class_target) * term2) / num_pos

        # loss for box_pred
        intersection = (tf.minimum(t, tp) + tf.minimum(b, bp)) * (tf.minimum(l, lp) + tf.minimum(r, rp))
        area_traget = (t + b) * (r + l)
        area_pred = (tp + bp) * (rp + lp)
        union = tf.maximum(1e-10, area_pred + area_traget - intersection)

        iou = tf.reduce_mean(intersection[not_empty_mask] / union[not_empty_mask])
        iou_loss = -tf.math.log(iou)
        # iou_loss = 1 - tf.reduce_mean(intersection[not_empty_mask] / union[not_empty_mask])

        centerness_loss = -tf.reduce_mean(centerness_target * tf.math.log(centerness_pred) +
                                          (1 - centerness_target) * tf.math.log(1 - centerness_pred))

        return  focal_loss, iou_loss, centerness_loss  # can add coefficients

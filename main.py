import tensorflow as tf
import argparse
from models.EDet import EDet
from time import time

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)

# tf.config.run_functions_eagerly(True)

data_dir = r'.\data'

# Parser
parser = argparse.ArgumentParser(description="Global parameters for training.")
# parser.add_argument("--train", type=int, default=1, help="whether to train the network")
parser.add_argument("--epochs", type=int, default=1000, help="epochs to train")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of optimizer")
parser.add_argument("--batch_size", type=int, default=4, help="train batch sizes")
parser.add_argument("--val_batch_size", type=int, default=4, help="val batch size")
# parser.add_argument("--test_video_len", type=int, default=200, help="length of test video")
parser.add_argument("--display_step", type=int, default=100,
                    help="number of steps we cycle through before displaying detailed progress")
parser.add_argument("--validation_step", type=int, default=100,
                    help="number of steps we cycle through before validating the model")
parser.add_argument("--summary_step", type=int, default=100,
                    help="number of steps we cycle through before saving summary")
parser.add_argument("--checkpoint_step", type=int, default=500,
                    help="number of steps we cycle through before saving checkpoint")
parser.add_argument("--base_dir", type=str, default="./results", help="directory in which results will be stored")
parser.add_argument("--data_dir", type=str, default=data_dir, help="data directory")
parser.add_argument("--max_to_keep", type=int, default=5, help="number of checkpoint files to keep")
parser.add_argument("--img_h", type=int, default=224, help="size of network input image")
parser.add_argument("--img_w", type=int, default=224, help="size of network input image")
parser.add_argument("--model_name", type=str, default="test", help="name of model")
# parser.add_argument("--show_summary", type=int, default=1, help="show summary of the model")
parser.add_argument("--start_step", type=int, default=0, help="starting step of training")
parser.add_argument("--verbose", type=int, default=1, help="verbosity of model")
parser.add_argument("--FPN_depth", type=int, default=1, help="FPN depth")
parser.add_argument("--FPN_conv_filters", type=int, default=16, help="FPN conv filters")
parser.add_argument("--clsbox_conv_filters", type=int, default=16, help="classnet and boxnet conv filters")
parser.add_argument("--classnet_convs_num", type=int, default=4, help="classnet convs")
parser.add_argument("--boxnet_convs_num", type=int, default=4, help="boxnet convs")
parser.add_argument('--speed', dest='speed_test', action='store_true')
parser.set_defaults(speed_test=False)
args = parser.parse_args()

# Main
if __name__ == "__main__":
    model = EDet(
        img_size=(args.img_h, args.img_w),
        FPN_depth=args.FPN_depth,
        FPN_conv_filters=args.FPN_depth,
        clsbox_conv_filters=args.clsbox_conv_filters,
        classnet_convs_num=args.classnet_convs_num,
        boxnet_convs_num=args.boxnet_convs_num,
        data_dir=args.data_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        base_dir=args.base_dir,
        training=not args.speed_test,
        max_to_keep=args.max_to_keep,
        model_name=args.model_name,
        start_step=args.start_step,
        verbose=args.verbose
    )
    model.ckpt_and_sum_setup()
    # test
    if args.speed_test:
        model.forward_pass(tf.ones((1, args.img_h, args.img_w, 3)), training=False)
        times = []
        for _ in range(100):
            x = tf.random.uniform((1, args.img_h, args.img_w, 3), 0, 1)
            t0 = time()
            model.forward_pass(x, training=False)
            t = time() - t0
            print('TIME:', t)
            times.append(t)
        print('\nMEAN:', sum(times) / len(times))
        exit()

    # train
    model.get_data_iterators()
    model.train_model(args.display_step, args.validation_step, args.checkpoint_step, args.summary_step)

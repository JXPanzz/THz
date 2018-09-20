import argparse
from glob import glob
import tensorflow as tf
from six.moves import xrange
from ops import *
from utils import *


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='seg', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size_H', dest='fine_size_H', type=int, default=256, help='then crop to this size')

parser.add_argument('--fine_size_w', dest='fine_size_W', type=int, default=256, help='then crop to this size')

parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=10, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.00014, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=600, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=200.0, help='weight on L1 term in objective')

parser.add_argument("--frozen_model_filename", default='model/frozen_model.pb',
                        type=str, help='Frozen model file to import')

args = parser.parse_args()

sample_files = glob('./datasets/{}/test/*.jpg'.format(args.dataset_name))
sample = [load_data(sample_file, is_test=True) for sample_file in sample_files]
sample_images = np.array(sample).astype(np.float32)
sample_images = [sample_images[i:i + args.batch_size]
                 for i in xrange(0, len(sample_images), args.batch_size)]

sample_images = np.array(sample_images)
images = sample_images[0, :, :, :, 0:3]

# def load_graph(frozen_graph_file):
#     with tf.gfile.GFile(frozen_graph_file, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def, name='prefix')
#         return graph


def load_graph(fz_gh_fn):
    with tf.gfile.GFile(fz_gh_fn, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name=""
            )
    return graph
graph = load_graph(args.frozen_model_filename)



if __name__ == "__main__":
    for op in graph.get_operations():
        print(op.name)
    iinput = graph.get_tensor_by_name('strided_slice:0')
    ooutput = graph.get_tensor_by_name('generator/Tanh:0')
    with tf.Session(graph=graph) as sess:

        ooutput = sess.run(ooutput, feed_dict={iinput: images})
        tmp = np.concatenate((images, ooutput), axis=2)
        save_images(tmp, [2, 1],
                    './{}/test_{:04d}.png'.format(args.test_dir, 00))
        print(ooutput.shape)
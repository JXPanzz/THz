import os, argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import matplotlib.pyplot as plt
from ops import *
from utils import *
from scipy.io import *
dir = os.path.dirname(os.path.realpath(__file__))


def freeze_graph(model_folder):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/Thz/model" + "/frozen_model.pb"

    # Before exporting our graph, we need to precise what is our output node
    # this variables is plural, because you can have multiple output nodes
    # freeze之前必须明确哪个是输出结点,也就是我们要得到推论结果的结点
    # 输出结点可以看我们模型的定义
    # 只有定义了输出结点,freeze才会把得到输出结点所必要的结点都保存下来,或者哪些结点可以丢弃
    # 所以,output_node_names必须根据不同的网络进行修改
    output_node_names = "fake_B"

    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
    clear_devices = True
    with tf.Session() as sess:
        # We import the meta graph and retrive a Saver
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
        saver.restore(sess, input_checkpoint)
        # We retrieve the protobuf graph definition
        # graph = tf.get_default_graph()
        input_graph_def = sess.graph.as_graph_def()

        for node in input_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in xrange(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
            # elif node.op == 'AssignAdd':
            #     node.op = 'Add'
            #     if 'use_locking' in node.attr:
            #         del node.attr['use_locking']

        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")  # We split on comma for convenience
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, default='model', help="Model folder to export")
    args = parser.parse_args()

    freeze_graph(args.model_folder)
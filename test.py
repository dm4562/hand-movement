from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow import TensorArray as ta
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import csv
import random
import os.path
import sys
import tarfile
import struct
import numpy as np
import random
import train

from datetime import datetime
from six.moves import urllib
from os import walk, getcwd, listdir
from os.path import basename


FLAGS = {
    'output_graph': '/data/muscle_rec/output_graph.pb',
    'root_image_dir': '/data/muscle_rec',
    'bottleneck_dir': '/data/muscle_rec/bottleneck',
    'final_tensor_name': 'final_result',
    'model_dir': '/data/muscle_rec/imagenet',
    'output_tensor_name': 'output_tensor'
}

# Constants required to load the pretrained CIFAR model
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

data_folders = ['hand1', 'hand2']


def get_data_labels(data_dir):
    """
    Gets all the images in the data_dir and returns all the filepaths alongwith
    the corresponding labels.

    Args:
        data_dir: The directory where the data is stored

    Returns:
        List[Tuple(str, str, float)]: The data to test on as a list of tuples 
        where a tuple contains (folder_path, image_name, label).
    """
    data_samples = []
    for folder in data_folders:
        image_path = "{}/images/{}".format(data_dir, folder)
        csv_path = "{}/normalized_{}.csv".format(data_dir, folder)

        samples = listdir(image_path)
        # samples = ["{}/{}".format(image_path, x) for x in sorted(samples)]
        samples = [(folder, x) for x in sorted(samples)]

        with open(csv_path, 'r') as file:
            reader = csv.reader(file)

            # Skip the header
            next(reader, None)

            for index, row in enumerate(reader):
                label = [float(x) for x in row[1::]]
                folder, image = samples[index]
                samples[index] = (folder, image, label)

        data_samples.extend(samples)

    random.shuffle(data_samples)
    return data_samples


def read_bottlenecks(bgraph, image_name, image_folder, jpeg_data_tensor,
                     bottleneck_tensor):
    """
    Retrieves the cached bottleneck values for an image.

    If a cached image does not exist on disk, recalculate the cached value and store
    it on disk for future use.

    Args:
        bgraph: The graph used to calculate the bottleneck values.
        image_name: Name of the image whose bottleneck is to be calculated.
        image_folder: The directory where the image whose bottleneck
        is to be calculated is located.
        jpeg_data_tensor: The tensor to feed the loaded jpeg into.
        bottleneck_tensor: The output tensor for the bottleneck values.

    Returns:
        Numpy array of values produced by the bottleneck layer for the image.
    """
    bottleneck_location = os.path.join(FLAGS['bottleneck_dir'], image_folder)
    train.ensure_dir_exists(bottleneck_location)
    bottleneck_path = os.path.join(
        FLAGS['bottleneck_dir'], image_folder, image_name)

    if not os.path.exists(bottleneck_path):
        with tf.Session(graph=bgraph) as sess:
            train.create_bottleneck_file(sess, bottleneck_path, image_name,
                                         image_folder, jpeg_data_tensor, bottleneck_tensor)

    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False

    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except:
        print("Invalid float found, recreating bottleneck")
        did_hit_error = True

    if did_hit_error:
        with tf.Session(graph=bgraph) as sess:
            train.create_bottleneck_file(sess, bottleneck_path, image_name,
                                         image_folder, jpeg_data_tensor, bottleneck_tensor)

        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()

        # Allow exceptions to propagate here, since they shouldn't happen after
        # a fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return np.array(bottleneck_values)


def load_saved_graph():
    """
    Creates a graph from the saved output GraphDef file and return a graph
    object.

    Returns:
        Graph holding the trained final tensor
        final trained weights tensor
    """
    with tf.Session() as sess:
        with gfile.FastGFile(FLAGS['output_graph'], 'rb') as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())

            final_weights_tensor = tf.import_graph_def(
                graph_def, name='', return_elements=[FLAGS['final_tensor_name']])
    return sess.graph, final_weights_tensor


def add_output_node(output_count, output_tensor_name, bottleneck_tensor, final_weights_tensor):
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_defaults(
            bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
            name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(tf.float32, shape=[None, output_count],
                                            name='GroundTruthInput')

        final_weights = tf.placeholder(tf.float32, shape=[BOTTLENECK_TENSOR_SIZE, output_count],
                                       name='TrainedWeightsPlaceholder')

    with tf.name_scope('ouput'):
        output = tf.matmul(bottleneck_input, final_weights,
                           name=output_tensor_name)

    return bottleneck_input, ground_truth_input, final_weights, output


def main(_):
    # Set up the graph to calculate bottlenecks
    train.maybe_download_and_extract()
    bgraph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = train.create_inception_graph()

    # Load the images to test on
    image_dict = get_data_labels(FLAGS['root_image_dir'])

    tgraph, weights = load_saved_graph()

    # Make sure all the images are cached
    with tf.Session() as sess:
        train.cache_bottlenecks(sess, image_dict, FLAGS[
            'bottleneck_dir'], jpeg_data_tensor, bottleneck_tensor)

    bottlenecks = []
    for folder, image, label in image_dict:
        bottlenecks.append(read_bottlenecks(
            bgraph, image, folder, jpeg_data_tensor, bottleneck_tensor))

    # print(bottlenecks)
    ground_truths = [label for _, _, label in image_dict]

    with tf.Session() as sess:
        bottleneck_input, ground_truth_input, output = add_output_node(
            5, FLAGS['output_tensor_name'], bottleneck_tensor, weights)

        a = sess.run([output], feed_dict={bottleneck_input: bottlenecks,
                                          ground_truth_input: ground_truths})
        print(a)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])

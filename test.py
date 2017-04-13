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
    'model_dir': '/data/muscle_rec/imagenet'
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

    return bottleneck_values


def load_saved_graph():
    """                                                           
    Creates a graph from the saved output GraphDef file and return a graph                                             │
    object.                                                       

    Returns:                                                      
        Graph holding the trained final tensor                    
        final trained weights tensor                              
    """
    with tf.Session() as sess:
        with gfile.FastGFile(FLAGS['output_graph'], 'rb') as file:
            graph_def = ts.GraphDef()
            graph_def.ParseFromString(file.read())

    final_weights_tensor = tf.import_graph_def(graph_def, name='',
                                               [FLAGS['final_tensor_name']])
    return sess.graph, final_weights_tensor


def main(_):
    # Set up the graph to calculate bottlenecks
    train.maybe_download_and_extract()
    bgraph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = train.create_inception_graph()

    # Load the images to test on
    image_dict = get_data_labels(FLAGS['root_image_dir'])

    # Make sure all the images are cached
    with tf.Session(graph=bgraph) as sess:
        train.cache_bottlenecks(sess, image_dict['test'], FLAGS[
            'bottleneck_dir'], jpeg_data_tensor, bottleneck_tensor)

    tgraph, output_tensor = load_saved_graph()

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])

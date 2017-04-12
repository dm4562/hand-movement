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

    # for sample in data_samples:
        # print(sample)

    # labels = [label for _, _, label in data_samples]
    # filenames = [f for f, _, _ in data_samples]
    # return filenames, labels
    random.shuffle(data_samples)
    data_dict = {
        'test': data_samples
    }

    return data_dict


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
            graph_def = ts.GraphDef()
            graph_def.ParseFromString(file.read())

    final_weights_tensor = tf.import_graph_def(graph_def, name='',
                                               [FLAGS['final_tensor_name']])

    return sess.graph, final_weights_tensor


def maybe_download_and_extract():
    """Download and extract model tar file.

    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.
    """
    dest_directory = FLAGS['model_dir']
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def create_inception_graph():
    """"Creates a graph from saved GraphDef file and returns a Graph object.

    Returns:
        Graph holding the trained Inception network, and various tensors we'll be
        manipulating.
    """
    with tf.Session() as sess:
        model_filename = os.path.join(
            FLAGS['model_dir'], 'classify_image_graph_def.pb')

        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))

    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def ensure_dir_exists(dir_name):
    """
    Makes sure the folder exists on disk.

    Args:
        dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_image_path(image_name, image_folder, image_dir=ROOT_IMAGE_DIRECTORY):
    return os.path.join(image_dir, 'images', image_folder, image_name)


def get_bottleneck_path(image_name, image_folder, bottleneck_dir=BOTTLENECK_DIRECTORY):
    return os.path.join(bottleneck_dir, image_folder, image_name)


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    """
    Runs inference on an image to extract the 'bottleneck' summary layer.

    Args:
        sess: Current active TensorFlow Session.
        image_data: String of raw JPEG data.
        image_data_tensor: Input data layer in the graph.
        bottleneck_tensor: Layer before the final softmax.

    Returns:
        Numpy array of bottleneck values.
    """
    bottleneck_values = sess.run(
        bottleneck_tensor,
        {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def create_bottleneck_file(sess, bottleneck_path, image_name, image_folder,
                           jpeg_data_tensor, bottleneck_tensor):
    print('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_name, image_folder)
    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)

    image_data = gfile.FastGFile(image_path, 'rb').read()
    bottleneck_values = run_bottleneck_on_image(
        sess, image_data, jpeg_data_tensor, bottleneck_tensor)

    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_name, image_folder, bottleneck_dir,
                             jpeg_data_tensor, bottleneck_tensor):
    """Retrieves or calculates bottleneck values for an image.

    If a cached version of the bottleneck data exists on-disk, return that,
    otherwise calculate the data and save it to disk for future use.

    Args:
        sess: The current active TensorFlow Session.
        image_lists: Dictionary of training images for each label.
        label_name: Label string we want to get an image for.
        index: Integer offset of the image we want. This will be modulo-ed by the
        available number of images for the label, so it can be arbitrarily large.
        image_dir: Root folder string  of the subfolders containing the training
        images.
        category: Name string of which  set to pull images from - training, testing,
        or validation.
        bottleneck_dir: Folder string holding cached files of bottleneck values.
        jpeg_data_tensor: The tensor to feed loaded jpeg data into.
        bottleneck_tensor: The output tensor for the bottleneck values.

    Returns:
        Numpy array of values produced by the bottleneck layer for the image.
    """
    # label_lists = image_lists[label_name]
    # sub_dir = label_lists['dir']
    image_folder_path = os.path.join(bottleneck_dir, image_folder)
    ensure_dir_exists(image_folder_path)
    bottleneck_path = get_bottleneck_path(image_name, image_folder)

    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(sess, bottleneck_path, image_name,
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
        create_bottleneck_file(sess, bottleneck_path, image_name,
                               image_folder, jpeg_data_tensor, bottleneck_tensor)

        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()

        # Allow exceptions to propagate here, since they shouldn't happen after
        # a fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values


def cache_bottlenecks(sess, image_list, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
    """Ensures all the training, testing, and validation bottlenecks are cached.

    Because we're likely to read the same image multiple times (if there are no
    distortions applied during training) it can speed things up a lot if we
    calculate the bottleneck layer values once for each image during
    preprocessing, and then just read those cached values repeatedly during
    training. Here we go through all the images we've found, calculate those
    values, and save them off.

    Args:
        sess: The current active TensorFlow Session.
        image_lists: Dictionary of training images for each label.
        image_dir: Root folder string of the subfolders containing the training
        images.
        bottleneck_dir: Folder string holding cached files of bottleneck values.
        jpeg_data_tensor: Input tensor for jpeg data from file.
        bottleneck_tensor: The penultimate output layer of the graph.

    Returns:
        Nothing.
    """
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)

    for image_folder, image, label in image_list:
        get_or_create_bottleneck(sess, image, image_folder, bottleneck_dir,
                                 jpeg_data_tensor, bottleneck_tensor)

        how_many_bottlenecks += 1
        if how_many_bottlenecks % 100 == 0:
            print(str(how_many_bottlenecks) +
                  ' bottleneck files created.')

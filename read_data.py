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

from six.moves import urllib
from os import walk, getcwd, listdir
from os.path import basename

# features = tf.placeholder(tf.int32, shape=[5], name='datapoints')
# timestamp = tf.placeholder(tf.int32, name='timestamp')
# total = tf.reduce_sum(features, name='total')
# printerop = tf.Print(total, [timestamp, features, total], name='printer')

# def create_file_reader(filename_queue):
#     reader = tf.TextLineReader(skip_header_lines=0)
#     _, row = reader.read(filename_queue)
#     defaults = [[0], [0], [0], [0], [0], [0]]
#     timestamp, point_1, point_2, point_3, point_4, point_5 = tf.decode_csv(row, record_defaults=defaults)
#     features = tf.stack([point_1, point_2, point_3, point_4, point_5])
#     return timestamp, features

# filenames = ['data/hand1.csv', 'data/hand2.csv']
# filename_queue = tf.train.string_input_producer(filenames)
# timestamp, features = create_file_reader(filename_queue)

# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)

#     for i in range(10):
#         time, points = sess.run([timestamp, features])
#         print(points)
#         # print(time)

# coord.request_stop()
# coord.join(threads)

# Progression -> read_data -> create_or_find_bottleneck -> train_last_layer

NUM_CHANNELS = 3
IMAGE_HEIGHT, IMAGE_WIDTH = 435, 336
BATCH_SIZE = 5

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
FLAGS = {
    'model_dir': '/tmp/imagenet',
    'final_tensor_name': 'final_result'
}

# test_size = int(len(filenames) * 0.2)
test_size = 5
data_folders = ['hand1', 'hand2']
ROOT_IMAGE_DIRECTORY = '/Users/dhruvmehra/Projects/tensorflow_stuff/research/data'
BOTTLENECK_DIRECTORY = '/tmp/bottleneck'

# Defined in retraining.py
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
# MODEL_INPUT_WIDTH = 299
# MODEL_INPUT_HEIGHT = 299
# MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


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
    return data_samples


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
    """Makes sure the folder exists on disk.

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
    """Runs inference on an image to extract the 'bottleneck' summary layer.

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


def cache_bottlenecks(sess, image_lists, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
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


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):
    """Adds a new softmax and fully-connected layer for training.

    We need to retrain the top layer to identify our new classes, so this function
    adds the right operations to the graph, along with some variables to hold the
    weights, and then sets up all the gradients for the backward pass.

    The set up for the softmax and fully-connected layers is based on:
    https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

    Args:
        class_count: Integer of how many categories of things we're trying to
        recognize.
        final_tensor_name: Name string for the new final node that produces results.
        bottleneck_tensor: The output of the main CNN graph.

    Returns:
        The tensors for the training and cross entropy results, and tensors for the
        bottleneck input and ground truth input.
    """
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
            name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(tf.float32,
                                            [None, class_count],
                                            name='GroundTruthInput')

    # Organizing the following ops as `final_training_ops` so they're easier
    # to see in TensorBoard
    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
        layer_weights = tf.Variable(tf.truncated_normal(
            [BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001), name='final_weights')
            variable_summaries(layer_weights)

        with tf.name_scope('biases'):
        layer_biases = tf.Variable(
            tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)

        with tf.name_scope('Wx_plus_b'):
            logits = tf.add(tf.matmul(bottleneck_input,
                                      layer_weights), layer_biases)
            tf.summary.histogram('pre_activations', logits)

    # final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    final_tensor = logits
    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cost'):
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        #     labels=ground_truth_input, logits=logits)
        # error = tf.pow(final_tensor - ground_truth_input, 2)

        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)

    tf.summary.scalar('cost', cross_entropy_mean)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(FLAGS['learning_rate']).minimize(
            cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
            final_tensor)


def main():
    # Set up the pre-trained graph.
    maybe_download_and_extract()
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = create_inception_graph()

    image_list = get_data_labels(getcwd() + '/data')
    sess = tf.Session()

    # Make sure that we've calculated the 'bottleneck' image summaries and cached them
    # on disk.
    cache_bottlenecks(sess, image_list, BOTTLENECK_DIRECTORY,
                      jpeg_data_tensor, bottleneck_tensor)

    # Add the new layer that we'll be training.
    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(len(image_lists.keys()),
                                            FLAGS['final_tensor_name'],
                                            bottleneck_tensor)

    # filenames, labels = get_data_labels(getcwd() + '/data')
    # all_images = ops.convert_to_tensor(filenames, dtype=dtypes.string)
    # all_labels = ops.convert_to_tensor(labels, dtype=dtypes.float32)

    # partitions = [0] * len(filenames)

    # print("Filenames: ", filenames)
    # print("labels: ", labels)

    # partitions[:test_size] = [1] * test_size
    # random.shuffle(partitions)

    # train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
    # train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)

    # train_input_queue = tf.train.slice_input_producer(
    #     [train_images, train_labels], shuffle=False)
    # test_input_queue = tf.train.slice_input_producer(
    #     [test_images, test_labels], shuffle=False)

    # # Process string tensor and label tensor into an image and label
    # file_content = tf.read_file(train_input_queue[0])
    # train_image = tf.image.decode_jpeg(file_content, NUM_CHANNELS)
    # train_label = train_input_queue[1]

    # file_content = tf.read_file(test_input_queue[0])
    # test_image = tf.image.decode_jpeg(file_content, NUM_CHANNELS)
    # test_label = test_input_queue[1]

    # train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    # test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

    # train_image_batch, train_label_batch = tf.train.batch(
    #     [train_image, train_label], batch_size=BATCH_SIZE)
    # test_image_batch, test_label_batch = tf.train.batch(
    #     [test_image, test_label], batch_size=BATCH_SIZE)

    # print("INPUT PIPELINE READY")

    # with tf.Session() as sess:
    #     # Initialize all variables
    #     # sess.run(tf.initialize_all_variables())
    #     tf.global_variables_initializer().run()

    #     # Initialize the queue threads to start the shovel data
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)

    #     print("from the train set")
    #     for i in range(20):
    #         # print(sess.run(train_label_batch))
    #         print(sess.run(train_image_batch))

    #     print("from the test set")
    #     for i in range(10):
    #         print(sess.run(test_label_batch))

    #     coord.request_stop()
    #     coord.join(threads=threads)
    #     sess.close()


if __name__ == '__main__':
    main()

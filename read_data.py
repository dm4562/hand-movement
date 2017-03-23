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

from six.moves import urllib
from os import walk, getcwd, listdir
from os.path import basename

NUM_CHANNELS = 3
IMAGE_HEIGHT, IMAGE_WIDTH = 435, 336
BATCH_SIZE = 5

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
FLAGS = {
    'model_dir': '/tmp/imagenet',
    'final_tensor_name': 'final_result',
    'summaries_dir': '/tmp/retrain_logs',
    'training_steps': 4000,
    'train_batch_size': 100,
    'validation_batch_size': 100,
    'test_batch_size': -1,
    'eval_step_interval': 10
}

data_folders = ['hand1', 'hand2']
ROOT_IMAGE_DIRECTORY = '/Users/dhruvmehra/Projects/tensorflow_stuff/research/data'
BOTTLENECK_DIRECTORY = '/tmp/bottleneck'

# Defined in retraining.py
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
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
    random.shuffle(data_samples)
    test_size = int(0.2 * len(data_samples))
    data_dict = {
        'test': data_samples[:test_size],
        'train': data_samples[2 * test_size:],
        'validate': data_samples[test_size: 2 * test_size]
    }
    return data_dict


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


def get_random_cached_bottlenecks(sess, image_dict, how_many, category, jpeg_data_tensor,
                                  bottleneck_tensor, bottleneck_dir=BOTTLENECK_DIRECTORY):
    """Retrieves bottleneck values for cached images.

    If no distortions are being applied, this function can retrieve the cached
    bottleneck values directly from disk for images. It picks a random set of
    images from the specified category.

    Args:
        sess: Current TensorFlow Session.
        image_lists: Dictionary of training images for each label.
        how_many: If positive, a random sample of this size will be chosen.
        If negative, all bottlenecks will be retrieved.
        category: Name string of which set to pull from - training, testing, or
        validation.
        bottleneck_dir: Folder string holding cached files of bottleneck values.
        image_dir: Root folder string of the subfolders containing the training
        images.
        jpeg_data_tensor: The layer to feed jpeg image data into.
        bottleneck_tensor: The bottleneck output layer of the CNN graph.

    Returns:
        List of bottleneck arrays, their corresponding ground truths, and the
        relevant filenames.
    """
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        # Retrieve a random sample of bottlenecks.
        for unused_i in range(how_many):
            image_index = random.randrange(len(image_dict[category]))
            image_folder, image_name, label = image_dict[category][image_index]
            image_path = get_image_path(image_name, image_folder)
            bottleneck = get_or_create_bottleneck(sess, image_name, image_folder,
                                                  BOTTLENECK_DIRECTORY, jpeg_data_tensor,
                                                  bottleneck_tensor)
            bottlenecks.append(bottleneck)
            ground_truths.append(label)
            filenames.append(image_path)
    else:
        # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(image_dict.keys()):
            for image_index, image_name in enumerate(image_dict[category]):
                image_path = get_image_path(image_name, image_folder)
                bottleneck = get_or_create_bottleneck(sess, image_name, image_folder,
                                                      BOTTLENECK_DIRECTORY, jpeg_data_tensor,
                                                      bottleneck_tensor)
                bottlenecks.append(bottleneck)
                ground_truths.append(label)
                filenames.append(image_path)

    return bottlenecks, ground_truths, filenames


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
        error = tf.sqrt(tf.pow(final_tensor - ground_truth_input, 2))

        with tf.name_scope('total'):
            error_mean = tf.reduce_mean(error)

    tf.summary.scalar('cost', error_mean)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(FLAGS['learning_rate']).minimize(
            error_mean)

    return (train_step, error_mean, bottleneck_input, ground_truth_input,
            final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.

    Args:
        result_tensor: The new final node that produces results.
        ground_truth_tensor: The node we feed ground truth data
        into.

    Returns:
        Tuple of (evaluation step, prediction).
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(
                prediction, tf.argmax(ground_truth_tensor, 1))

        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def main(_):
    # Set up the pre-trained graph.
    maybe_download_and_extract()
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = create_inception_graph()

    image_dict = get_data_labels(ROOT_IMAGE_DIRECTORY)
    sess = tf.Session()

    # Make sure that we've calculated the 'bottleneck' image summaries and cached them
    # on disk.
    assert(len(image_dict['train']) > len(image_dict['test']))
    assert(len(image_dict['test']) == len(image_dict['validate']))

    cache_bottlenecks(sess, image_dict['train'] + image_dict['test'] + image_dict['validate'],
                      BOTTLENECK_DIRECTORY, jpeg_data_tensor, bottleneck_tensor)

    # Add the new layer that we'll be training.
    (train_step, error_mean, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(len(image_lists.keys()),
                                            FLAGS['final_tensor_name'],
                                            bottleneck_tensor)

    # Create the operations we need to evaluate the accuracy of our new layer.
    evaluation_step, prediction = add_evaluation_step(
        final_tensor, ground_truth_input)

    # Merge all the summaries and write them out to /tmp/retrain_logs (by
    # default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS['summaries_dir'] + '/train',
                                         sess.graph)
    validation_writer = tf.summary.FileWriter(
        FLAGS['summaries_dir'] + '/validation')

    # Set up all our weights to their initial default values.
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(FLAGS['training_steps']):
        # Get a batch of input bottleneck values, either calculated fresh every time
        # with distortions applied, or from the cache stored on disk.

        train_bottlenecks, train_ground_truth, _ = get_random_cached_bottlenecks(
            sess, image_dict, FLAGS['train_batch_size'], 'train',
            jpeg_data_tensor, bottleneck_tensor)

        # Feed the bottlenecks and ground truth into the graph, and run a training
        # step. Capture training summaries for TensorBoard with the `merged`
        # op.
        train_summary, _ = sess.run([merged, train_step],
                                    feed_dict={bottleneck_input: train_bottlenecks,
                                               ground_truth_input: train_ground_truth})
        train_writer.add_summary(train_summary, i)

        # Every so often, print out how well the graph is training.
        is_last_step = (i + 1 == FLAGS.how_many_training_steps)
        if (i % FLAGS['eval_step_interval']) == 0 or is_last_step:
            train_accuracy, cross_entropy_value = sess.run(
                [evaluation_step, error_mean], feed_dict={bottleneck_input: train_bottlenecks,
                                                          ground_truth_input: train_ground_truth})

            print('%s: Step %d: Train accuracy = %.1f%%' %
                  (datetime.now(), i, train_accuracy * 100))
            print('%s: Step %d: Cross entropy = %f' %
                  (datetime.now(), i, cross_entropy_value))

            validation_bottlenecks, validation_ground_truth, _ = get_random_cached_bottlenecks(
                sess, image_dict, FLAGS['validation_batch_size'], 'validate',
                jpeg_data_tensor, bottleneck_tensor)

            # Run a validation step and capture training summaries for TensorBoard
            # with the `merged` op.
            validation_summary, validation_accuracy = sess.run(
                [merged, evaluation_step], feed_dict={bottleneck_input: validation_bottlenecks,
                                                      ground_truth_input: validation_ground_truth})
            validation_writer.add_summary(validation_summary, i)
            print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                  (datetime.now(), i, validation_accuracy * 100,
                   len(validation_bottlenecks)))

    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.
    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(sess, image_lists, FLAGS.test_batch_size,
                                      'testing', FLAGS.bottleneck_dir,
                                      FLAGS.image_dir, jpeg_data_tensor,
                                      bottleneck_tensor))
    test_accuracy, predictions = sess.run([evaluation_step, prediction],
                                          feed_dict={bottleneck_input: test_bottlenecks,
                                                     ground_truth_input: test_ground_truth})
    print('Final test accuracy = %.1f%% (N=%d)' % (
        test_accuracy * 100, len(test_bottlenecks)))

    # Write out the trained graph and labels with the weights stored as
    # constants.
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FLAGS.final_tensor_name])

    with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())

    with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
        f.write('\n'.join(image_lists.keys()) + '\n')

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])

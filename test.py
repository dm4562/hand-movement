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

from train import cache_bottlenecks, maybe_download_and_extract, create_inception_graph

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


def main(_):
    # Set up the graph to calculate bottlenecks
    maybe_download_and_extract()
    bgraph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = create_inception_graph()

    # Load the images to test on
    image_dict = get_data_labels(FLAGS['root_image_dir'])
    sess = tf.Session(graph=bgraph)

    cache_bottlenecks(sess, image_dict['test'], FLAGS[
                      'bottleneck_dir'], jpeg_data_tensor, bottleneck_tensor)

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])

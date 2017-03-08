import tensorflow as tf
from tensorflow import TensorArray as ta
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import csv
import random
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

# test_size = int(len(filenames) * 0.2)
test_size = 5


def split_test_train(data_samples):
    train, test = [], []

    for (i, point) in data_samples:
        i = i % 10

        if i < 7:
            train.append(point)
        else:
            test.append(point)

    return (train, test)


def get_data_labels(data_dir):
    data_files = ['hand1', 'hand2']
    data_samples = []
    for file in data_files:
        image_path = "{}/images/{}".format(data_dir, file)
        csv_path = "{}/normalized_{}.csv".format(data_dir, file)

        samples = listdir(image_path)
        samples = ["{}/{}".format(image_path, x) for x in sorted(samples)]

        with open(csv_path, mode='r') as file:
            reader = csv.reader(file)

            # Skip the header
            next(reader, None)

            for index, row in enumerate(reader):
                label = [float(x) for x in row[1::]]
                samples[index] = (samples[index], label)

        data_samples.extend(samples)

    # for sample in data_samples:
    #     print(sample)

    labels = [label for _, label in data_samples[:20]]
    filenames = [f for f, _ in data_samples[:20]]
    return filenames, labels

if __name__ == '__main__':
    filenames, labels = get_data_labels(getcwd() + '/data')
    all_images = ops.convert_to_tensor(filenames, dtype=dtypes.string)
    all_labels = ops.convert_to_tensor(labels, dtype=dtypes.float32)

    partitions = [0] * len(filenames)

    partitions[:test_size] = [1] * test_size
    random.shuffle(partitions)

    train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
    train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)

    train_input_queue = tf.train.slice_input_producer(
        [train_images, train_labels], shuffle=False)
    test_input_queue = tf.train.slice_input_producer(
        [test_images, test_labels], shuffle=False)

    # Process string tensor and label tensor into an image and label
    file_content = tf.read_file(train_input_queue[0])
    train_image = tf.image.decode_jpeg(file_content, NUM_CHANNELS)
    train_label = train_input_queue[1]

    file_content = tf.read_file(test_input_queue[0])
    test_image = tf.image.decode_jpeg(file_content, NUM_CHANNELS)
    test_label = test_input_queue[1]

    train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

    train_image_batch, train_label_batch = tf.train.batch(
        [train_image, train_label], batch_size=BATCH_SIZE)
    test_image_batch, test_label_batch = tf.train.batch(
        [test_image, test_label], batch_size=BATCH_SIZE)

    print("INPUT PIPELINE READY")

    with tf.Session() as sess:
        # Initialize all variables
        # sess.run(tf.initialize_all_variables())
        tf.global_variables_initializer().run()

        # Initialize the queue threads to start the shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print("from the train set")
        for i in range(20):
            print(sess.run(train_label_batch))

        print("from the test set")
        for i in range(10):
            print(sess.run(test_label_batch))

        coord.request_stop()
        coord.join(threads=threads)
        sess.close()

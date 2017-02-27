import tensorflow as tf
from tensorflow import TensorArray as ta
import csv

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


def get_sample_paths(data_dir):
    data_files = ['hand1', 'hand2']
    data_samples = []
    for file in data_files:
        image_path = "{}/images/{}".format(data_dir, file)
        csv_path = "{}/{}.csv".format(data_dir, file)

        samples = listdir(image_path)
        samples = ["{}/{}".format(image_path, x) for x in sorted(samples)]

        with open(csv_path, mode='r') as file:
            reader = csv.reader(file)
            for index, row in enumerate(reader):
                label = [int(x) for x in row[1::]]
                samples[index] = (samples[index], label)

        data_samples.append(samples)

    return data_samples

if __name__ == '__main__':
    get_image_names(getcwd() + '/data')

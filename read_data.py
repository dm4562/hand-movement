import tensorflow as tf
from tensorflow import TensorArray as ta

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

def get_image_names(data_dir):
    data_files = ['hand1']

    for file in data_files:
        path = "{}/images/{}".format(data_dir, file)

        samples = listdir(path)
        samples = ["{}/{}".format(path, x) for x in sorted(samples)]
        for i in range(50):
            print(samples[i])

if __name__ == '__main__':
    get_image_names(getcwd() + '/data')
    
        
import tensorflow as tf
from tensorflow import TensorArray as ta

features = tf.placeholder(tf.int32, shape=[5], name='datapoints')
timestamp = tf.placeholder(tf.int32, name='timestamp')
total = tf.reduce_sum(features, name='total')
printerop = tf.Print(total, [timestamp, features, total], name='printer')

def create_file_reader(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=0)
    _, row = reader.read(filename_queue)
    defaults = [[0], [0], [0], [0], [0], [0]]
    timestamp, point_1, point_2, point_3, point_4, point_5 = tf.decode_csv(row, record_defaults=defaults)
    features = tf.stack([point_1, point_2, point_3, point_4, point_5])
    return timestamp, features

filenames = ['data/hand1.csv', 'data/hand2.csv']
filename_queue = tf.train.string_input_producer(filenames)
timestamp, features = create_file_reader(filename_queue)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10):
        time, points = sess.run([timestamp, features])
        print(points)
        # print(time)

coord.request_stop()
coord.join(threads)

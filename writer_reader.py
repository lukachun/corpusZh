import tensorflow as tf
import numpy

def write_binary():
    writer = tf.python_io.TFRecordWriter('data.tfrecord')

    for i in range(0, 2):
        a = 0.618 + i
        b = [2016 + i, 2017+i]
        c = numpy.array([[0, 1, 2],[3, 4, 5]]) + i
        c = c.astype(numpy.uint8)
        c_raw = c.tostring()

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'a': tf.train.Feature(
                        float_list=tf.train.FloatList(value=[a])
                    ),

                    'b': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=b)
                    ),
                    'c': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[c_raw])
                    )
                }
            )
        )
        serialized = example.SerializeToString()
        writer.write(serialized)

    writer.close()

def read_single_sample(filename):
    # output file name string to a queue
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)

    # create a reader from file queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # get feature from serialized example

    features = tf.parse_single_example(
        serialized_example,
        features={
            'a': tf.FixedLenFeature([], tf.float32),
            'b': tf.FixedLenFeature([2], tf.int64),
            'c': tf.FixedLenFeature([], tf.string)
        }
    )

    a = features['a']

    b = features['b']

    c_raw = features['c']
    c = tf.decode_raw(c_raw, tf.uint8)
    c = tf.reshape(c, [2, 3])

    return a, b, c

#-----main function-----
if 0:
    write_binary()
else:
    # create tensor
    a, b, c = read_single_sample('data.tfrecord')
    a_batch, b_batch, c_batch = tf.train.shuffle_batch([a, b, c], batch_size=3, capacity=200, min_after_dequeue=100, num_threads=2)

    queues = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)

    # sess
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)
    a_val, b_val, c_val = sess.run([a_batch, b_batch, c_batch])
    print(a_val, b_val, c_val)
    a_val, b_val, c_val = sess.run([a_batch, b_batch, c_batch])
    print(a_val, b_val, c_val)

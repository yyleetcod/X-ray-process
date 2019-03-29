
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def _parse_function(record):
    """
    Extract data from a `tf.Example` protocol buffer.
    """
    # Defaults are not specified since both keys are required.
    keys_to_features = {
        'label': tf.FixedLenFeature([8], tf.float32),
        'image': tf.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)

    # Extract features from single example
    # image_decoded = tf.image.decode_image(parsed_features['image'])
    image_decoded = tf.decode_raw(parsed_features['image'], tf.uint8)
    image_reshaped = tf.reshape(
        image_decoded, [500, 500, 3])
    points = tf.cast(parsed_features['label'], tf.float32)
    '''
    with tf.Session() as sess:
        print (sess.run(points))
    '''
    return image_reshaped, points


def input_fn(record_file, batch_size, num_epochs=None, shuffle=True):
    """
    Input function required for TensorFlow Estimator.
    """
    dataset = tf.data.TFRecordDataset(record_file)

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(_parse_function)
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=10000)
    if batch_size != 1:
        dataset = dataset.batch(batch_size)
    if num_epochs != 1:
        dataset = dataset.repeat(num_epochs)

    # Make dataset iteratable.
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    feature, label = iterator.get_next()
    with tf.Session() as sess:
        print (sess.run(label))
        print (sess.run(feature).shape)
        print (sess.run(feature)[1][203][200])
    return feature, label

if __name__ == '__main__':
    input_fn('./train.tfrecords', 5)
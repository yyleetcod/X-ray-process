'''
Resnet18 (with some change in full connected layer) Estimator for X-ray image classification.
'''
import numpy as np
import tensorflow as tf
import time

IMG_WIDTH = 600
IMG_HEIGHT = 600
IMG_CHANNEL = 1
L2_RATE = 25.0
BATCH_SIZE = 32
TRAINING_STEP = 2000
USE_FOCAL = True

def focal_loss(labels, logits, gamma=2):
    y_pred = tf.nn.softmax(logits, dim=-1) # [BATCH_SIZE,num_classes]
    labels = tf.one_hot(labels, depth=y_pred.shape[1])
    loss = -labels * ((1 - y_pred) ** gamma) * tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0))
    loss = tf.reduce_sum(loss)
    return loss


def resnet_block(inputs, num_filters, kernel_size, strides, activation='relu'):
    x = tf.layers.Conv2D(num_filters, kernel_size=kernel_size, strides=strides, 
        padding='same', kernel_initializer='he_normal',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_RATE))(inputs)
    x = tf.layers.BatchNormalization()(x)
    if(activation):
        x = tf.nn.relu(x)
    return x


def cnn_model_fn(features, labels, mode):
    """The model function for the network."""
    # Input feature x should be of shape (BATCH_SIZE, image_width, image_height, color_channels).
    # Image shape should be checked for safety reasons at early stages, and could be removed
    # before training actually starts.
    # assert features['x'].shape[1:] == (
    #     IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL), 'Image size does not match.'
    # Concat front image and side image for input image 
    img = tf.concat([features['side_img'], features['front_img']], -1)
    inputs = 2 * (tf.to_float(img, name='input_to_float') / 256) - 1

    # conv1
    x = resnet_block(inputs, 64, [7, 7], 2)

    # conv2
    x = tf.layers.MaxPooling2D([3, 3], 2, 'same')(x)
    for i in range(2):
        a = resnet_block(x, 64, [3, 3], 1)
        b = resnet_block(a, 64, [3,3], 1, activation=None)
        x = tf.math.add(x, b)
        x = tf.nn.relu(x)
    
    # conv3
    a = resnet_block(x, 128, [1, 1], 2)
    b = resnet_block(a, 128, [3, 3], 1, activation=None)
    x = tf.layers.Conv2D(128, kernel_size=[1, 1],strides=2, padding = 'same',
        kernel_initializer='he_normal', kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_RATE))(x)
    x = tf.math.add(x, b)
    x = tf.nn.relu(x)

    a = resnet_block(x, 128, [3, 3], 1)
    b = resnet_block(a, 128, [3, 3], 1, activation=None)
    #x=tf.keras.layers.add([x,b])
    x = tf.math.add(x, b)
    x = tf.nn.relu(x)

    # conv4
    a = resnet_block(x, 256, [1, 1], 2)
    b = resnet_block(a, 256, [3, 3], 1, activation=None)
    x = tf.layers.Conv2D(256, kernel_size=[1, 1], strides=2, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_RATE))(x)
    #x=tf.keras.layers.add([x,b])
    x = tf.math.add(x, b)
    x = tf.nn.relu(x)

    a = resnet_block(x, 256, [3, 3], 1)
    b = resnet_block(a, 256, [3, 3], 1, activation=None)
    #x=tf.keras.layers.add([x,b])
    x = tf.math.add(x, b)
    x = tf.nn.relu(x)

    # conv5
    a = resnet_block(x, 512, [1, 1], 2)
    b = resnet_block(a, 512, [3, 3], 1, activation=None)
    x = tf.layers.Conv2D(512, kernel_size=[1, 1], strides=2, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_RATE))(x)
    #x=tf.keras.layers.add([x,b])
    x = tf.math.add(x, b)
    x = tf.nn.relu(x)

    a = resnet_block(x, 512, [3, 3], 1)
    b = resnet_block(a, 512, [3, 3], 1, activation=None)
    #x=tf.keras.layers.add([x,b])
    x = tf.math.add(x, b)
    x = tf.nn.relu(x)
    x = tf.layers.AveragePooling2D(pool_size=10, strides=3, data_format='channels_last')(x)

    # Flatten tensor into a batch of vectors
    y = tf.layers.Flatten()(x)

    # Use sex, weight and height infomation to help model predict more accuracy
    sex = features['sex'] # tf.expand_dims(features['sex'], 1)
    sex = tf.one_hot(sex, 2, 1.0, 0.0, dtype=tf.float32)
    height = tf.expand_dims(features['height'], 1)
    weight = tf.expand_dims(features['weight'], 1)
    more_info = tf.concat([sex, height, weight], -1)
    # Pass a dense layer to make those information has a dimension comparable with feature map
    more_info = tf.layers.Dense(512, kernel_initializer='he_normal',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_RATE))(more_info)
    more_info = tf.nn.relu(more_info)
    y = tf.concat([y, more_info], -1)
    # y = tf.concat([y, sex, height, weight], -1)
    y = tf.layers.Dense(1024, kernel_initializer='he_normal', 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_RATE))(y)
    y = tf.nn.relu(y)
    logits = tf.layers.Dense(17, kernel_initializer='he_normal', 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_RATE))(y)
    
    # Split predict logits, the first 8 for thigh, the rest 9 for shin
    logits_thigh = logits[:, :8]
    logits_shin = logits[:, 8:]

    # Prediction output
    probabilities_thigh = tf.nn.softmax(logits_thigh)
    predicted_thigh_classes = tf.argmax(logits_thigh, 1)
    predict_thigh_val_top3, predict_thigh_index_top3 = tf.nn.top_k(probabilities_thigh, k=3)

    probabilities_shin = tf.nn.softmax(logits_shin)
    predicted_shin_classes = tf.argmax(logits_shin, 1)
    predict_shin_val_top3, predict_shin_index_top3 = tf.nn.top_k(probabilities_shin, k=3)
    # Make prediction for PREDICATION mode.
    predictions_dict = {
        # 'probabilities_thigh': probabilities_thigh,
        'Index of Picture': features['index'],
        'Top3 Class of Thigh Bone': predict_thigh_index_top3,
        'Top3 Probability of Thigh Bone': predict_thigh_val_top3,
        'Label of Thigh Bone': features['thigh_bone'],
        # 'probabilities_shin': probabilities_shin,
        'Top3 Class of Shin Bone': predict_shin_index_top3,
        'Top3 Probability of Shin Bone': predict_shin_val_top3,
        'Label of Shin Bone': features['shin_bone']
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict)

    label_thigh_tensor = tf.convert_to_tensor(labels['thigh_bone'], dtype=tf.int64)
    label_shin_tensor = tf.convert_to_tensor(labels['shin_bone'], dtype=tf.int64)
    # Caculate loss using mean squared error.
    # Sparse softmaxcrossentropy
   
    if USE_FOCAL:
        # Focal loss
        loss_thigh = focal_loss(label_thigh_tensor, logits_thigh)
        # pre_thigh_label = label_thigh_tensor - tf.cast(label_thigh_tensor > 0, tf.int64)
        # next_thigh_label = label_thigh_tensor + tf.cast(label_thigh_tensor < 7, tf.int64)
        # loss_thigh += 0.1 * focal_loss(pre_thigh_label, logits_thigh) + 0.1 * focal_loss(next_thigh_label, logits_thigh)
        loss_shin = focal_loss(label_shin_tensor, logits_shin)
        # pre_shin_label = label_shin_tensor - tf.cast(label_shin_tensor > 0, tf.int64)
        # next_shin_label = label_shin_tensor + tf.cast(label_shin_tensor < 8, tf.int64)
        # loss_shin += 0.1 * focal_loss(pre_shin_label, logits_shin) + 0.1 * focal_loss(next_shin_label, logits_shin)
    else:
        # Sparse softmaxcrossentropy
        loss_thigh = tf.losses.sparse_softmax_cross_entropy(
                labels=label_thigh_tensor, logits=logits_thigh)
        loss_shin = tf.losses.sparse_softmax_cross_entropy(
                labels=label_shin_tensor, logits=logits_shin)

    loss = loss_thigh + loss_shin
    # Regularization loss
    reg_loss = tf.losses.get_total_loss()
    loss += reg_loss

    # Configure the train OP for TRAIN mode.
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_steps = tf.train.get_global_step()
        # Decay learning rate
        # learning_rate = tf.train.exponential_decay(0.0001, global_steps, 10, 0.9, staircase=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            export_outputs={'marks': tf.estimator.export.RegressionOutput(logits)})

    # Add evaluation metrics (for EVAL mode)
    accuracy_thigh = tf.metrics.accuracy(
            labels=label_thigh_tensor,
            predictions=predicted_thigh_classes)
    accuracy_in_top_3_thigh = tf.metrics.mean(
        tf.cast(tf.nn.in_top_k(probabilities_thigh, label_thigh_tensor, 3), tf.float32))
    accuracy_shin = tf.metrics.accuracy(
            labels=label_shin_tensor,
            predictions=predicted_shin_classes)
    accuracy_in_top_3_shin = tf.metrics.mean(
        tf.cast(tf.nn.in_top_k(probabilities_shin, label_shin_tensor, 3), tf.float32))
    eval_metric_ops = {
            'Loss of Thigh Bone': tf.metrics.mean(loss_thigh),
            'Loss of Shin Bone': tf.metrics.mean(loss_shin),
            'Accuracy of Thigh Bone': accuracy_thigh,
            'Top3 Accuracy of Thigh Bone': accuracy_in_top_3_thigh,
            'Accuracy of Shin Bone': accuracy_shin,
            'Top3 Accuracy of Shin Bone': accuracy_in_top_3_shin}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def _parse_function(record):
    """Extract data from a `tf.Example` protocol buffer."""
    keys_to_features = {
        'index': tf.FixedLenFeature([], tf.int64), 
        'front_img': tf.FixedLenFeature([], tf.string),
        'side_img': tf.FixedLenFeature([], tf.string),
        'sex': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.float32),
        'weight': tf.FixedLenFeature([], tf.float32),
        'direction': tf.FixedLenFeature([], tf.int64),
        'thigh_bone': tf.FixedLenFeature([], tf.int64),
        'shin_bone': tf.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)

    # Extract features from single example
    # Front image
    front_image_decoded = tf.decode_raw(parsed_features['front_img'], tf.uint8)
    front_image_reshaped = tf.reshape(
        front_image_decoded, [IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL])
    # Side image
    side_image_decoded = tf.decode_raw(parsed_features['side_img'], tf.uint8)
    side_image_reshaped = tf.reshape(
        side_image_decoded, [IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL])
    
    features = {'index': parsed_features['index'],
            'front_img': front_image_reshaped,
            'side_img': side_image_reshaped,
            'sex': parsed_features['sex'],
            'height': parsed_features['height'],
            'weight': parsed_features['weight'],
            'direction': parsed_features['direction'],
            'thigh_bone': parsed_features['thigh_bone'], # just for prediction ouput, DO NOT use it for training
            'shin_bone': parsed_features['shin_bone'] # just for prediction ouput, DO NOT use it for training
            }
    labels = {'thigh_bone':parsed_features['thigh_bone'],
            'shin_bone':parsed_features['shin_bone']}

    return features, labels


def train_input_fn(record_file, BATCH_SIZE):
    """Input function required for TensorFlow Estimator."""
    dataset = tf.data.TFRecordDataset(record_file)

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(1000).repeat().batch(BATCH_SIZE)

    # Make dataset iteratable.
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    features, labels = iterator.get_next()
    return features, labels


def eval_and_predict_input_fn(record_file, BATCH_SIZE, mod='eval'):
    """Input function required for TensorFlow Estimator."""
    dataset = tf.data.TFRecordDataset(record_file)

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(_parse_function)
    
    dataset = dataset.batch(BATCH_SIZE)
    
    # Make dataset iteratable.
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    features, labels = iterator.get_next()
    
    if mod == 'predict':
        return features
    else:
        return features, labels 


def main(unused_argv):
    # Create the Estimator
    """
    # gpu config
    session_config = tf.ConfigProto(
        log_device_placement=True, 
        inter_op_parallelism_threads=0, 
        intra_op_parallelism_threads=0, 
        allow_soft_placement=True,
        device_count={'GPU': 0})
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.99
    run_config = tf.estimator.RunConfig().replace(session_config=session_config)
    """

    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='./train') #, config=run_config)

    # Choose mode between Train, Evaluate and Predict
    mode_dict = {
        'train': tf.estimator.ModeKeys.TRAIN,
        'eval': tf.estimator.ModeKeys.EVAL,
        'predict': tf.estimator.ModeKeys.PREDICT
    }

    # Training stage
    print('\nStart to train model')
    start = time.time()
    estimator.train(
            input_fn=lambda: train_input_fn('./train_data.tfrecords', BATCH_SIZE), steps=TRAINING_STEP)
    end = time.time()
    print('End to train model')
    print('-' * 100)
    print(f'Training Time: {(end - start):.3f}sec!')
    print('-' * 100)

    # Export result as SavedModel.
    # estimator.export_savedmodel('./saved_model', serving_input_receiver_fn)

    # Evaluating stage
    print('\nStart to evaluate model in training set')
    start = time.time()
    evaluation = estimator.evaluate(
        input_fn=lambda: eval_and_predict_input_fn('./train_data.tfrecords', BATCH_SIZE))
    end = time.time()
    print('End to evaluate model in training set')
    print('-' * 100)
    print(f'Eval time in training set: {(end - start):.3f}sec!')
    for key in evaluation:
        print('{}\t{}'.format(key, evaluation[key]))
    print('-' * 100)

    print('\nStart to evaluate model in validation set')
    start = time.time()
    evaluation = estimator.evaluate(input_fn=lambda: eval_and_predict_input_fn('./val_data.tfrecords', BATCH_SIZE))
    end = time.time()
    print('End to evaluate model in validation set')
    print('-' * 100)
    print(f'Eval Time in validation set: {(end - start):.3f}sec!')
    for key in evaluation:
        print('{}\t{}'.format(key, evaluation[key]))
    print('-' * 100)
    
    predictions = estimator.predict(
        input_fn=lambda: eval_and_predict_input_fn('./val_data.tfrecords', BATCH_SIZE, 'predict'))
    # Print 10 of predictions
    print ('\nInformation of the Predictions')
    for index, result in enumerate(predictions):
        print('-' * 100)
        for key in result:
            print ('key: {}\t value: {}'.format(key, result[key]))
    print('-' * 100)

if __name__ == '__main__':
    tf.app.run()

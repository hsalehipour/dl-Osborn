
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# hyper parameters
hparams = {
    'learning_rate': 1e-3,
    'dropout_rate': 0.5,
    'activation': tf.nn.elu
}

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""

    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info("model_fn: PREDICT, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info("model_fn: EVAL, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("model_fn: TRAIN, {}".format(mode))

    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # 3 vertical profiles are measured at 512 points and have one channel
    input_layer = tf.reshape(features["x"], [-1, 2, 512, 1])
    input_layer_batch_normalized = tf.layers.batch_normalization(input_layer, axis=2)

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer_batch_normalized,
        filters=64,
        kernel_size=[2, 4],
        strides=(1, 1),
        padding="same",
        activation=hparams["activation"])
    conv1_dropout = tf.layers.dropout(inputs=conv1, rate=hparams["dropout_rate"], training=mode == tf.estimator.ModeKeys.TRAIN)

    # Pooling Layer #1
    pool1 = tf.layers.average_pooling2d(inputs=conv1_dropout, pool_size=[1, 2], strides=[1, 2])

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=48,
        kernel_size=[2, 4],
        strides=(1, 1),
        padding="same",
        activation=hparams["activation"])
    conv2_dropout = tf.layers.dropout(inputs=conv2, rate=hparams["dropout_rate"], training=mode == tf.estimator.ModeKeys.TRAIN)

    # Pooling Layer #2
    # pool2 = tf.layers.average_pooling2d(inputs=conv2_dropout, pool_size=[1, 2], strides=[1, 2])

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=conv2_dropout,
        filters=48,
        kernel_size=[2, 4],
        strides=(1, 1),
        padding="same",
        activation=hparams["activation"])
    conv3_dropout = tf.layers.dropout(inputs=conv3, rate=hparams["dropout_rate"], training=mode == tf.estimator.ModeKeys.TRAIN)

    # Pooling Layer #3
    # pool3 = tf.layers.average_pooling2d(inputs=conv3_dropout, pool_size=[1, 2], strides=[1, 2])

    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=conv3_dropout,
        filters=24,
        kernel_size=[2, 4],
        strides=(1, 1),
        padding="same",
        activation=hparams["activation"])
    conv4_dropout = tf.layers.dropout(inputs=conv4, rate=hparams["dropout_rate"], training=mode == tf.estimator.ModeKeys.TRAIN)

    # Pooling Layer #4
    pool4 = tf.layers.average_pooling2d(inputs=conv4_dropout, pool_size=[1, 2], strides=[1, 2])

    # Convolutional Layer #5
    conv5 = tf.layers.conv2d(
        inputs=pool4,
        filters=20,
        kernel_size=[2, 4],
        strides=(1, 1),
        padding="same",
        activation=hparams["activation"])
    conv5_dropout = tf.layers.dropout(inputs=conv5, rate=hparams["dropout_rate"], training=mode == tf.estimator.ModeKeys.TRAIN)

    # Pooling Layer #5
    pool5 = tf.layers.average_pooling2d(inputs=conv5_dropout, pool_size=[1, 2], strides=[1, 2])

    # Convolutional Layer #6
    conv6 = tf.layers.conv2d(
        inputs=pool5,
        filters=16,
        kernel_size=[2, 4],
        strides=(1, 1),
        padding="same",
        activation=hparams["activation"])
    conv6_dropout = tf.layers.dropout(inputs=conv6, rate=hparams["dropout_rate"], training=mode == tf.estimator.ModeKeys.TRAIN)

    # Pooling Layer #6
    pool6 = tf.layers.average_pooling2d(inputs=conv6_dropout, pool_size=[1, 2], strides=[1, 2])

    #
    # # Convolutional Layer #7
    # conv7 = tf.layers.conv2d(
    #     inputs=pool6,
    #     filters=8,
    #     kernel_size=[2, 2],
    #     strides=(1, 1),
    #     padding="same",
    #     activation=hparams["activation"])
    # conv7_dropout = tf.layers.dropout(inputs=conv7, rate=hparams["dropout_rate"], training=mode == tf.estimator.ModeKeys.TRAIN)
    #
    # # Pooling Layer #6
    # pool7 = tf.layers.average_pooling2d(inputs=conv7_dropout, pool_size=[1, 2], strides=[1, 2])
    #
    #
    # # Convolutional Layer #8
    # conv8 = tf.layers.conv2d(
    #     inputs=pool7,
    #     filters=4,
    #     kernel_size=[2, 2],
    #     strides=(1, 1),
    #     padding="same",
    #     activation=hparams["activation"])
    # conv8_dropout = tf.layers.dropout(inputs=conv8, rate=hparams["dropout_rate"], training=mode == tf.estimator.ModeKeys.TRAIN)
    #
    # # Pooling Layer #8
    # pool8 = tf.layers.average_pooling2d(inputs=conv8_dropout, pool_size=[1, 2], strides=[1, 2])
    #
    # # Convolutional Layer #9
    # conv9 = tf.layers.conv2d(
    #     inputs=pool8,
    #     filters=1,
    #     kernel_size=[2, 2],
    #     strides=(1, 1),
    #     padding="same",
    #     activation=tf.nn.sigmoid)
    # conv9_dropout = tf.layers.dropout(inputs=conv9, rate=hparams["dropout_rate"], training=mode == tf.estimator.ModeKeys.TRAIN)
    #
    # # Pooling Layer #9
    # pool9 = tf.layers.average_pooling2d(inputs=conv9_dropout, pool_size=[2, 2], strides=[2, 2])
    #
    # # Output layer
    # regressed_output = tf.squeeze(pool9)


    # Flatten tensor into a batch of vectors
    pool6_flat = tf.reshape(pool6, [-1, 2 * 32 * 16])

    # Dense Layer
    dense = tf.layers.dense(inputs=pool6_flat, units=64, activation=hparams["activation"])

    # Add dropout operation
    dropout = tf.layers.dropout(
        inputs=dense, rate=hparams["dropout_rate"], training=mode == tf.estimator.ModeKeys.TRAIN)


    # Output layer
    regressed_output = tf.squeeze(tf.layers.dense(inputs=dropout, units=1, activation=tf.nn.sigmoid))

    # Generate predictions (for PREDICT and EVAL mode)
    predictions = {"efficiency": regressed_output}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions["efficiency"])
    tf.summary.scalar('Loss', loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=hparams["learning_rate"])
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "error_rmse": tf.metrics.root_mean_squared_error(
            labels=labels, predictions=predictions["efficiency"]),
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def dnn_model_fn(features, labels, mode):
    """Model function for CNN."""

    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info("model_fn: PREDICT, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info("model_fn: EVAL, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("model_fn: TRAIN, {}".format(mode))

    # batch normalize input layer
    # input_layer_reshaped = tf.reshape(features["x"], [-1, 3, 512])
    # input_layer_batch_normalized = tf.layers.batch_normalization(input_layer_reshaped, axis=-1)
    # input_layer = tf.reshape(input_layer_batch_normalized, [-1, 3 * 512])
    input_layer = features["x"]

    # Dense Layer #1
    dense1 = tf.layers.dense(inputs=input_layer, units=4096, activation=tf.nn.elu)

    # Dense Layer #2
    dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.elu)

    # Dense Layer #3
    dense3 = tf.layers.dense(inputs=dense2, units=128, activation=tf.nn.elu)

    # Dense Layer #4
    dense4 = tf.layers.dense(inputs=dense3, units=32, activation=tf.nn.elu)

    # Add dropout operation
    dropout = tf.layers.dropout(
        inputs=dense4, rate=hparams["dropout_rate"], training=mode == tf.estimator.ModeKeys.TRAIN)

    # Output layer
    regressed_output = tf.squeeze(tf.layers.dense(inputs=dropout, units=1, activation=tf.nn.sigmoid))

    # Generate predictions (for PREDICT and EVAL mode)
    predictions = {"efficiency": regressed_output}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions["efficiency"])
    tf.summary.scalar('Loss', loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=hparams["learning_rate"])
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "error_rmse": tf.metrics.root_mean_squared_error(
            labels=labels, predictions=predictions["efficiency"]),
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


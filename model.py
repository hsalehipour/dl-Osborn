
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
from dataset import save_data


# hyper parameters
hparams = {
    'learning_rate': 1e-3,
    'dropout_rate': 0.5,
    'activation': tf.nn.relu
}


def set_flags():
    # DEFAULT SETTINGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir' , type=str, default='./experiments/test/r1/', help='Directory that stores all training logs and trained models')
    parser.add_argument('--train_data', type=str, default='./data/training_data.dat', help='Dataset for training')
    parser.add_argument('--test_data' , type=str, default='./data/testing_data.dat', help='Dataset for testing')
    parser.add_argument('--mode', type=str, default='train', help='Network MODE: "train", "eval", "infer" (predict or serving).  [default: "train"]')
    parser.add_argument('--training_steps', type=int, default=2e4, help='Number of training steps [default: 20,000]')
    parser.add_argument('--batch', type=int, default=100, help='Batch Size per GPU during training [default: 100]')
    parser.add_argument('--epoch', type=int, default=50, help='Epoch to run [default: 50]')
    FLAGS = parser.parse_args()
    return FLAGS


def optimizer_fn(lr):
    return tf.train.AdamOptimizer(learning_rate=lr, use_locking=True)


def loss_fn(labels, nn_output):
    loss = tf.losses.mean_squared_error(labels=labels, predictions=nn_output)
    return loss


def ConvNet(input, mode):
    """
    the graph of a convolutional neural network
    """

    # Determine whether mode is training
    istraining = mode == tf.estimator.ModeKeys.TRAIN

    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # 3 vertical profiles are measured at 512 points and have one channel
    input_layer = tf.reshape(input["x"], [-1, 2, 512, 1])
    input_layer_batch_normalized = tf.layers.batch_normalization(input_layer, axis=2)

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer_batch_normalized,
        filters=64,
        kernel_size=[2, 4],
        strides=(1, 1),
        padding="same",
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1234),
        activation=hparams["activation"])

    # Pooling Layer #1
    pool1 = tf.layers.average_pooling2d(inputs=conv1, pool_size=[1, 2], strides=[1, 2])

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[2, 4],
        strides=(1, 1),
        padding="same",
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1234),
        activation=hparams["activation"])

    # Pooling Layer #2
    pool2 = tf.layers.average_pooling2d(inputs=conv2, pool_size=[1, 2], strides=[1, 2])

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[2, 4],
        strides=(1, 1),
        padding="same",
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1234),
        activation=hparams["activation"])

    # Pooling Layer #3
    pool3 = tf.layers.average_pooling2d(inputs=conv3, pool_size=[1, 2], strides=[1, 2])

    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=64,
        kernel_size=[2, 4],
        strides=(1, 1),
        padding="same",
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1234),
        activation=hparams["activation"])

    # Pooling Layer #4
    pool4 = tf.layers.average_pooling2d(inputs=conv4, pool_size=[1, 2], strides=[1, 2])

    # Convolutional Layer #5
    conv5 = tf.layers.conv2d(
        inputs=pool4,
        filters=64,
        kernel_size=[2, 4],
        strides=(1, 1),
        padding="same",
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1234),
        activation=hparams["activation"])

    # Pooling Layer #5
    pool5 = tf.layers.average_pooling2d(inputs=conv5, pool_size=[1, 2], strides=[1, 2])

    # Convolutional Layer #6
    conv6 = tf.layers.conv2d(
        inputs=pool5,
        filters=64,
        kernel_size=[2, 4],
        strides=(1, 1),
        padding="same",
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1234),
        activation=hparams["activation"])

    # Pooling Layer #6
    pool6 = tf.layers.average_pooling2d(inputs=conv6, pool_size=[1, 2], strides=[1, 2])

    # # Convolutional Layer #7
    # conv7 = tf.layers.conv2d(
    #     inputs=pool6,
    #     filters=64,
    #     kernel_size=[2, 4],
    #     strides=(1, 1),
    #     padding="same",
    #     kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1234),
    #     activation=hparams["activation"])
    #
    # # Pooling Layer #7
    # pool7 = tf.layers.average_pooling2d(inputs=conv7, pool_size=[1, 2], strides=[1, 2])

    # Flatten tensor into a batch of vectors
    pool6_flat = tf.reshape(pool6, [-1, 2 * 64 * 8])

    # Dense Layer
    dense = tf.layers.dense(inputs=pool6_flat, units=64, activation=hparams["activation"])

    # Add dropout operation
    dropout = tf.layers.dropout(
        inputs=dense, rate=hparams["dropout_rate"], training=istraining, seed=1234)

    # Output layer
    output  = tf.squeeze(tf.layers.dense(inputs=dropout, units=1, activation=tf.nn.sigmoid))
    return output, conv1



def FCNet(features, mode):
    """
    Graph of a fully connected network.
    """

    # Input Layer
    # batch normalize input layer
    # input_layer_reshaped = tf.reshape(features["x"], [-1, 2, 512])
    # input_layer_batch_normalized = tf.layers.batch_normalization(input_layer_reshaped, axis=-1)
    # input_layer = tf.reshape(input_layer_batch_normalized, [-1, 2 * 512])
    input_layer = tf.reshape(features["x"], [-1, 2 * 512])

    # Dense Layer #1
    dense1 = tf.layers.dense(inputs=input_layer, units=128, activation=hparams["activation"])

    # Dense Layer #2
    dense2 = tf.layers.dense(inputs=dense1, units=128, activation=hparams["activation"])

    # Dense Layer #3
    dense3 = tf.layers.dense(inputs=dense2, units=128, activation=hparams["activation"])

    # Dense Layer #4
    dense4 = tf.layers.dense(inputs=dense3, units=256, activation=hparams["activation"])

    # Dense Layer #5
    dense5 = tf.layers.dense(inputs=dense4, units=256, activation=hparams["activation"])

    # Dense Layer #6
    dense6 = tf.layers.dense(inputs=dense5, units=128, activation=hparams["activation"])

    # Add dropout operation
    dropout = tf.layers.dropout(
        inputs=dense6, rate=hparams["dropout_rate"], training=mode == tf.estimator.ModeKeys.TRAIN)

    # Output layer
    output = tf.squeeze(tf.layers.dense(inputs=dropout, units=1, activation=tf.nn.sigmoid))

    return output




def model_fn(features, labels, mode, nn_graph = None):
    """
    Model function
    """

    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info("model_fn: PREDICT, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info("model_fn: EVAL, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("model_fn: TRAIN, {}".format(mode))

    # Find network output
    nn_output, nn_layer1 = nn_graph(features, mode)

    # Generate predictions (for PREDICT and EVAL mode)
    predictions_dic = {"efficiency": nn_output}

    # Set up a session hook for evaluating first layer outputs
    fetches_hook = FetchHook(nn_layer1)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = loss_fn(labels, nn_output)
        tf.summary.scalar('Loss', loss)

        # Output number of parameters
        nvar_graph = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
        tf.summary.scalar('Total_num_trainable_var', nvar_graph)

        optimizer = optimizer_fn(hparams["learning_rate"])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        # Add evaluation metrics (for EVAL mode)
        loss = loss_fn(labels, nn_output)
        tf.summary.scalar('Loss', loss)
        eval_metric_ops = {
            "error_rmse": tf.metrics.root_mean_squared_error(
                labels=labels, predictions=nn_output),
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          eval_metric_ops=eval_metric_ops, evaluation_hooks=[fetches_hook])
    else:
        # mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dic)




def cnn_model_fn(features, labels, mode):
    return model_fn(features, labels, mode, nn_graph=ConvNet)


def dnn_model_fn(features, labels, mode):
    return model_fn(features, labels, mode, nn_graph=FCNet)


class FetchHook(tf.train.SessionRunHook):

    def __init__(self, fetches):
        self.fetches = fetches
        return

    def begin(self):
        # You can add ops to the graph here.
        print('Starting the session.')

    def after_create_session(self, session, coord):
        # When this is called, the graph is finalized and
        # ops can no longer be added to the graph.
        print('Session created.')

    def before_run(self, run_context):
        print('Before calling session.run().')
        return tf.train.SessionRunArgs(self.fetches)

    def after_run(self, run_context, run_values):
        # 'F' fortran ordering because (unfortunately at this point) I need to read this output file in MATLAB!
        save_data({'conv_output': run_values.results.flatten('F')}, 'conv_output.dat')
        pass

    def end(self, session):
      print('Done with the session.')






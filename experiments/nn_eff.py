#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pylab as plt

# path to training directory
# TRAIN_DATA_DIR = "/home/hesam/workspace/osborn/training_data_features_chi_eps_N2.dat"
# TEST_DATA_DIR = "/home/hesam/workspace/osborn/prediction_data_features_chi_eps_N2.dat"
TRAIN_DATA_DIR = "/home/hesam/workspace/osborn/training_data_normalized_features_chi_eps_N2.dat"
TEST_DATA_DIR = "/home/hesam/workspace/osborn/prediction_data_normalized_features_chi_eps_N2.dat"
MODEL_DIR = "/home/hesam/workspace/osborn/cnn/test/"

# number of z-points in the z-profiles used for training
headers_feature = ['chi', 'eps', 'N2', 'eff']
nz_profile = 512
nfeatures = 2

def lrelu(x, alpha=0.001):
    """Leaky Relu activation function"""
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

# hyper parameters
hparams = {
    'learning_rate': 1e-3,
    'dropout_rate': 0.5,
    'activation': tf.nn.elu
}

def add_noise(data, nrand = 10):
    nexample = np.int32(data.shape[0]/nz_profile)
    for ie in range(nexample):
        istart = ie * nz_profile
        iend = istart + nz_profile-1
        profiles = data.loc[istart:iend,:]
        for i in range(nrand):
            data.append(add_rand_logn_noise(profiles))
    return data


def add_gaussian_noise(data, coefficient=0.1, mu=0.0, sigma=0.5):
    """
   :param data: Data to add Gaussian noise to.
   :param coefficient: Noise factor.
   :param mu: Mean of the distribution.
   :param sigma: Standard deviation of the distribution.
   :return: Noisy copy of the input data.
   """
    return data + coefficient * np.random.normal(loc=mu, scale=sigma, size=data.shape)


def add_rand_logn_noise(data, amp=0.01):
    """
    add random lognormal noise
    """
    m = np.mean(data, axis=0)
    v = np.var(data , axis=0)
    mu = np.log((m**2)/np.sqrt(v+m**2))
    sigma = np.sqrt(np.log(v/(m**2)+1.))
    return data + amp * np.random.lognormal(mean=mu, sigma=sigma, size=data.shape)


def r2_score(label, prediction):
    """
    calculate the R2 score of a regression
    """
    total_error = np.sum((label-np.mean(label))**2)
    residual_error = np.sum((label-prediction)**2)
    r2 = 1.0 - residual_error/(total_error + np.finfo(np.float).eps)

    return r2


def load_data(fdir=TRAIN_DATA_DIR, split=True, ratio=0.85):
    """
    Returns the mixing data-set as (train_x, train_y), (test_x, test_y). where _x and _y denote "features" and "labels"
    respectively.
    """
    data = pd.read_csv(fdir, names=headers_feature)
    eff = data.pop('eff')[0:data.shape[0]:nz_profile]
    chi = data.pop('chi')

    # # normalize the input features
    # data = normalize_data(data, axis=0)
    # xmin = data['eps'].min()
    # xmax = data['eps'].max()
    # data['eps']= (data['eps'] - xmin) / (xmax-xmin)
    # data['N2'] = (data['N2']  - xmin) / (xmax-xmin)

    # data must be reshaped
    features = np.array(data, dtype=np.float32).reshape((-1, nz_profile, nfeatures)).transpose((0,2,1))
    labels = np.array(eff, dtype=np.float32)

    # normalize the input features
    # features = normalize_data(features, axis=2)

    if split:
        # divide dataset (85-15) for training-testing
        train_x, test_x = split_data_train_test(features, ratio)
        train_y, test_y = split_data_train_test(labels  , ratio)
        return (train_x, train_y), (test_x, test_y)
    else:
        return features, labels

def normalize_data(x, axis=0):
    """"
    Normalizes the data to range between 0-1
    """
    x = np.array(x, dtype=np.float32)
    xmin = np.expand_dims(x.min(axis=axis), axis=axis)
    xmax = np.expand_dims(x.max(axis=axis), axis=axis)

    return (x-xmin)/(xmax-xmin)


def split_data_train_test(data, ratio):
    """
    splits feature or label data into "train" and "test"
    """
    # divide dataset (85-15) for training-testing
    nexample = data.shape[0]
    nexample_train = np.int32(ratio * nexample)

    # split data-set into training and testing set
    train_set = data[:nexample_train]
    test_set = data[nexample_train:]

    return train_set, test_set


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


def main(unused_argv):
    # Load training and eval data in np.array
    (train_data, train_labels), (eval_data, eval_labels) = load_data(TRAIN_DATA_DIR, split=True, ratio=0.85)
    # (train_data2, train_labels2), (eval_data2, eval_labels2) = load_data(TEST_DATA_DIR, split=True, ratio=0.0)
    # train_data   = np.append(train_data  , train_data2  , axis=0)
    # train_labels = np.append(train_labels, train_labels2, axis=0)
    # eval_data    = np.append(eval_data  , eval_data2    , axis=0)
    # eval_labels  = np.append(eval_labels, eval_labels2  , axis=0)


    # Create the Estimator
    osborn_nn_model = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=MODEL_DIR)

    # Set up logging for predictions
    # Log the values in a tensor with their labels using tensor_to_log dictionary
    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    osborn_nn_model.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    num_epochs = 20
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=num_epochs,
        shuffle=False)
    eval_results = osborn_nn_model.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    # Evaluate the model and print results
    pred_data, pred_labels = load_data(TEST_DATA_DIR, split=False)
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": pred_data},
        y=pred_labels,
        num_epochs=num_epochs,
        shuffle=False)
    pred_results = osborn_nn_model.evaluate(input_fn=pred_input_fn)
    print(pred_results)

    # Evaluate KHI test results
    predicted_KHI_results = osborn_nn_model.predict(input_fn=eval_input_fn)
    eff_KHI_predicted = np.array([prediction['efficiency'] for prediction in predicted_KHI_results]).reshape((num_epochs,eval_labels.size)).mean(axis=0)

    # Predict HWI results and compare with the true values
    predicted_HWI_results = osborn_nn_model.predict(input_fn=pred_input_fn)
    eff_HWI_predicted = np.array([prediction['efficiency'] for prediction in predicted_HWI_results]).reshape((num_epochs,pred_labels.size)).mean(axis=0)

    print({"Eval R2-Score" : r2_score(eval_labels, eff_KHI_predicted)})
    print({"Pred R2-Score" : r2_score(pred_labels, eff_HWI_predicted)})

    plt.figure(figsize=(10, 8))
    plt.plot(eff_KHI_predicted, 'r.-', eval_labels, 'b.-')
    plt.figure(figsize=(10, 8))
    plt.plot(eff_HWI_predicted, 'r.-', pred_labels, 'b.-')
    plt.show()


if __name__ == "__main__":
    tf.set_random_seed(1234)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

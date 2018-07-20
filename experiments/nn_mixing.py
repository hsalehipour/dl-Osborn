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



# path to training directory
TRAIN_DATA_DIR = "/home/hesam/workspace/osborn/training_data.dat"
MODEL_DIR = "/home/hesam/workspace/osborn/dnn/r3/"

# number of z-points in the z-profiles used for training
headers_feature    = ['N2','chi', 'mixing', 'dissipation']
nz_profile = 512
nfeatures = 2

# hyper parameters
hparams = {
    'learning_rate': 1e-3,
    'dropout_rate': 0.4,
    'activation'  : tf.nn.elu
}

def add_gaussian_noise(data, coefficient=0.1, mu=0.0, sigma=0.5):
   """
   :param data: Data to add Gaussian noise to.
   :param coefficient: Noise factor.
   :param mu: Mean of the distribution.
   :param sigma: Standard deviation of the distribution.
   :return: Noisy copy of the input data.
   """
   return data + coefficient * np.random.normal(loc=mu, scale=sigma, size=data.shape)




def load_data():
    """
    Returns the mixing data-set as (train_x, train_y), (test_x, test_y). where _x and _y denote "features" and "labels"
    respectively.
    """
    data = pd.read_csv(TRAIN_DATA_DIR, names=headers_feature)
    mixing  = data.pop('mixing')[0:data.shape[0]:nz_profile]

    # data must be reshaped
    features = np.array(data[['N2', 'chi']], dtype=np.float32).reshape((-1, nz_profile*nfeatures))
    labels   = np.array(mixing , dtype=np.float32)

    train_x, test_x = split_data_train_test(features)
    train_y, test_y = split_data_train_test(labels)

    return (train_x, train_y), (test_x, test_y)


def split_data_train_test(data):
    """
    splits feature or label data into "train" and "test"
    """
    # divide dataset (85-15) for training-testing
    nexample = data.shape[0]
    nexample_train = np.int32(0.85*nexample)

    # split data-set into training and testing set
    train_set = data[:nexample_train]
    test_set  = data[nexample_train:]

    return train_set, test_set


def load_dissipation():
    """
    Reads dissipation data
    """
    data = pd.read_csv(TRAIN_DATA_DIR, names=headers_feature)
    eps = data['dissipation'][0:data.shape[0]:nz_profile]
    return np.array(eps, dtype=np.float32)


def calc_eff(mixing, dissipation):
    """
    Calculates mixing efficiency
    """
    return  mixing/(mixing+dissipation)

def normalize(data):
    """
    normalzies the data to be between 0-1
    """
    return (data-data.min())/(data.max()-data.min())


def dnn_model_fn(features, labels, mode):
  """Model function for CNN."""

  if mode == tf.estimator.ModeKeys.PREDICT:
      tf.logging.info("model_fn: PREDICT, {}".format(mode))
  elif mode == tf.estimator.ModeKeys.EVAL:
      tf.logging.info("model_fn: EVAL, {}".format(mode))
  elif mode == tf.estimator.ModeKeys.TRAIN:
      tf.logging.info("model_fn: TRAIN, {}".format(mode))

  # batch normalize input layer
  input_layer = tf.reshape(features["x"], [-1, 2, 512])
  input_layer_batch_normalized = tf.layers.batch_normalization(input_layer, axis=2)
  input_layer_concat = tf.reshape(input_layer_batch_normalized, [-1, 2*512])
  # input_layer = features["x"]

  # Dense Layer #1
  dense1 = tf.layers.dense(inputs=input_layer_concat, units=4096, activation=None)

  # # Dense Layer #2
  # dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.elu)
  #
  # # Dense Layer #3
  # dense3 = tf.layers.dense(inputs=dense2, units=128, activation=tf.nn.elu)
  #
  # # Dense Layer #4
  # dense4 = tf.layers.dense(inputs=dense3, units=32, activation=tf.nn.elu)

  # Add dropout operation
  # dropout = tf.layers.dropout(
  #     inputs=dense1_batch_normalized, rate=hparams["dropout_rate"], training=mode == tf.estimator.ModeKeys.TRAIN)
  # dropout_batch_normalized = tf.layers.batch_normalization(dropout)

  # Output layer
  regressed_output = tf.squeeze(tf.layers.dense(inputs=dense1, units=1, activation=None))


  # Generate predictions (for PREDICT and EVAL mode)
  predictions = {"mixing": regressed_output}

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions["mixing"])
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
          labels=labels, predictions=predictions["mixing"]),
      }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def main(unused_argv):

  # Load training and eval data in np.array
  (train_data, train_labels), (eval_data, eval_labels) = load_data()

  # Create the Estimator
  osborn_convnet_model = tf.estimator.Estimator(
      model_fn=dnn_model_fn,
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
  osborn_convnet_model.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = osborn_convnet_model.evaluate(input_fn=eval_input_fn)
  print(eval_results)


  import matplotlib.pylab as plt
  predicted_eval_results = osborn_convnet_model.predict(input_fn=eval_input_fn)
  mixing_pred = np.array([prediction['mixing'] for prediction in predicted_eval_results])
  eps_train, eps_eval = split_data_train_test(load_dissipation())
  eff_eval = calc_eff(eval_labels, eps_eval)
  eff_pred = calc_eff(mixing_pred, eps_eval)

  plt.plot(eff_pred,'ro',eff_eval,'bo')
  # plt.plot(mixing_pred,'ro',eval_labels,'bo')
  plt.show()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

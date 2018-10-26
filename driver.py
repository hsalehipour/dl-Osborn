
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import model
import dataset
from utils import r2_score, rmse
import matplotlib.pylab as plt

# The set all the flags for running the network in various modes.
FLAGS = model.set_flags()


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    buffer_size = 1000
    random_seed = 1234

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(buffer_size, seed=random_seed).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def train(estimator_obj, train_data, train_labels, nsteps = FLAGS.training_steps, tensors_to_log = {}):
    """
    train the network as defined inside the estimator_obj with the given data
    :param estimator_obj: an instance of tf.estimator.Estimator
    :param train_data:   input values of the training dataset
    :param train_labels: labels of training dataset
    :param tensors_to_log: optional logging dictionary
    :return: None. The trained model is stored inside estimator_obj.model_dir
    """
    # define the training function
    # input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": train_data},
    #     y=train_labels,
    #     batch_size=FLAGS.batch,
    #     num_epochs=None,
    #     shuffle=True)
    input_fn = lambda: train_input_fn({"x": train_data}, train_labels, FLAGS.batch)

    # Set up logging for predictions
    # Log the values in a tensor with their labels using tensor_to_log dictionary
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    estimator_obj.train(
        input_fn=input_fn,
        steps=nsteps,
        hooks=[logging_hook])
    return


def evaluate(estimator_obj, data, labels, batch_size=FLAGS.batch, num_epochs=FLAGS.epoch):
    """
    Evaluates the model as defined inside estimator_obj
    """
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": data},
        y= labels,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=False)
    eval_results = estimator_obj.evaluate(input_fn=eval_input_fn)
    return eval_results

def predict(estimator_obj, input_data, num_epochs = FLAGS.epoch):
    """
    The trained model is served for prediction, as defined inside estimator_obj
    """
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": input_data},
        num_epochs=num_epochs,
        shuffle=False)
    results = estimator_obj.predict(input_fn=input_fn)
    return np.array([rdic['efficiency'] for rdic in results]).reshape((num_epochs, input_data.shape[0])).mean(axis=0)


def main(argv):

    # Create the Estimator
    config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir, tf_random_seed=1234)
    osborn_nn_model = tf.estimator.Estimator(model_fn=model.cnn_model_fn, config=config)

    # train the model
    if FLAGS.mode == 'train':
        # Load training and eval data in np.array
        # data_dir = FLAGS.train_data
        (train_data, train_labels) = dataset.load_data(FLAGS.train_data, ratio=1.0)
        train(osborn_nn_model, train_data, train_labels)

    # Evaluate the model and print results
    if FLAGS.mode == 'eval':
        data_dir = FLAGS.test_data
        (eval_data, eval_labels) = dataset.load_data(data_dir, ratio=1.0)
        eval_results = evaluate(osborn_nn_model, eval_data, eval_labels)
        print(eval_results)

    # Test the model and print results
    if FLAGS.mode == 'infer':
        data_dir = FLAGS.test_data
        pred_data, pred_labels = dataset.load_data(data_dir)

        # Predict HWI results and compare with the true values
        pred_results = predict(osborn_nn_model, pred_data)
        dataset.save_data({'eff_nn': pred_results}, 'dl_output.dat')
        print({"Pred R2-Score": r2_score(pred_labels, pred_results)})
        print({"Pred error_rmse": rmse(pred_labels, pred_results)})
        plt.figure(figsize=(10, 8))
        plt.plot(pred_results, 'r.-', pred_labels, 'b.-')
        plt.show()



if __name__ == "__main__":
    np.random.rand(1234)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

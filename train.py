
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import model
import dataset
from utils import r2_score

# path to training directory
# TRAIN_DATA_DIR = "/home/hesam/workspace/osborn/training_data_features_chi_eps_N2.dat"
# TEST_DATA_DIR = "/home/hesam/workspace/osborn/prediction_data_features_chi_eps_N2.dat"
TRAIN_DATA_DIR = "./data/training_data_normalized_features_chi_eps_N2.dat"
TEST_DATA_DIR = "./data/prediction_data_normalized_features_chi_eps_N2.dat"
MODEL_DIR = "./experiments/test/"

# number of z-points in the z-profiles used for training
headers_feature = ['chi', 'eps', 'N2', 'eff']
nz_profile = 512
nfeatures = 2





def main(unused_argv):
    # Load training and eval data in np.array
    (train_data, train_labels), (eval_data, eval_labels) = dataset.load_data(TRAIN_DATA_DIR, split=True, ratio=0.85)
    # (train_data2, train_labels2), (eval_data2, eval_labels2) = load_data(TEST_DATA_DIR, split=True, ratio=0.0)
    # train_data   = np.append(train_data  , train_data2  , axis=0)
    # train_labels = np.append(train_labels, train_labels2, axis=0)
    # eval_data    = np.append(eval_data  , eval_data2    , axis=0)
    # eval_labels  = np.append(eval_labels, eval_labels2  , axis=0)


    # Create the Estimator
    osborn_nn_model = tf.estimator.Estimator(model_fn = model.cnn_model_fn,model_dir= MODEL_DIR)

    # Set up logging for predictions
    # Log the values in a tensor with their labels using tensor_to_log dictionary
    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

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
    pred_data, pred_labels = dataset.load_data(TEST_DATA_DIR, split=False)
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Application logic

# https://www.tensorflow.org/tutorials/layers
# The MNIST dataset comprises 60,000 training examples and 10,000 test examples of the handwritten digits 0–9,
# formatted as 28x28-pixel monochrome images. Use layers to build a convolutional neural network model to
# recognize the handwritten digits in the MNIST data set.


def cnn_model_fn(features, labels, mode):
    """Model function for CNN"""
    # Input layer
    # Convert feature map to shape:
    # batch_size = -1 => dimension dynamically computed based on # of input values in features["x"]
    # image_width and image_height = 28x28 pixel images
    # channels = 1 =>  3 for red-green-blue, 1 for monochrome
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolution layer #1
    # Apply 32 5x5 filters to the input layer with a ReLU activation function,
    # using conv2d() to create this layer
    # kernel_size => dimension of the filters as [width, height]
    # padding = valid/same => output tensor should have the same width and height as the input ten%Ssor
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling layer #1
    # Connect first pooling layer to the created convolutional layer
    # Perform max pooling with a 2x2 filter and stride of 2 using max_pooling2d()
    # pool_size => size of the max pooling filter as [width, height]
    # strides => size of the strides, subregions extracted bt the filter should separate by 2 pixels in both width and height dimensions
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolution layer #2
    # Connect a second convolutional layer to the CNN: 64 5x5 filters with ReLU activation
    # Takes the output tensor of the first pooling layer as input
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling layer #2
    # Connect a second pooling layer to the CNN: 2x2 max pooling filter with a stride of 2
    # Takes the output tensor of the second convolutional layer as input
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense layer
    # Add a dense layer with 1024 neurons and ReLU activation to the CNN
    # to perform classification on features extracted by the convolution and pooling layers.
    # Flatten feature map to shape [batch_size, features] to the tensor has only 2 dimensions
    # Each example has 7 (pool2 width) * 7 (pool2 height) * 64 (pool2 channels) features
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # Connect the dense layer
    # units => numbers of neurons in the dense layer
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # Apply dropout regularization to the dense layer to improve the results of the model
    # rate => dropout rate, 40% of the elements will be randomly dropped out during training
    # training => dropout will be performend only in TRAIN mode
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Create dense layer with 10 neurons (one for each target class 0-9) with default linear activation
    # Returns the raw values for the predictions
    logits = tf.layers.dense(inputs=dropout, units=10)

    # Convert the raw values into two different formats that the model function can return
    # predicted class for each example = 0-9
    # probabilities for each target class for each example
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        # Predicted class is the element in the corresponding row of the logits tensor with the highest raw value
        # axis => axis of the input tensor along which to find the greatest value
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Compile the predictions into a dict and return an EstimatorSpec object
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # For training and evaluation, measure how closely the model's predictions match the target classes
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    # Configure the model to optimize the loss value during training with learning rate and optimization algorithm
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss, global_steps=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    # Add accuracy metric to the model
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # Training feature data (the raw pixel values for 55,000 images of hand-drawn digits)
    train_data = mnist.train.images  # Returns np.array
    # Training labels (the corresponding value from 0–9 for each image)
    trains_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # Evaluation feature data (10,000 images)
    eval_data = mnist.test.images  # Returns np.array
    # Evaluation labels
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    # model_fn => the model function to use for training, evaluation and prediction
    # model_dir => directory where the model data (checkpoints) will be saved
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    # Set up logging for predictions
    # CNNs can take some time to train, log the probability values from the softmax layer
    # of the CNN to track progress during training
    tensors_to_log = {"probabilities": "softmax_tensor"}
    # tensors => dict of tensors we want to log
    # every_n_iter => probabilities logged after every 50 steps of training
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    # x => training feature data
    # y => training labels
    # batch_size => the model will train on minibatches of 100 examples at each step
    # num_epoch => the model will train until the specifid steps is reached
    # shuffle => shuffle the training data
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=trains_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    # steps => the model will train for 20000 steps total
    # hooks => trigger the logging hook during training
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    # Determine the accuracy on the MNIST test set
    # num_epochs => the model evaluates the metrics over one epoch of data and returns the result
    # shuffle => iterate through the data sequentially
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()

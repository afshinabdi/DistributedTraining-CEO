"""
   Basic convolutional neural network for classification of MNIST data.
   The default is Lenet-5 like structure, two convolutional layers, followed by two fully connected ones.
   The filters' shapes are:
   [5, 5, 1, 32], [5, 5, 32, 64], [7 * 7 * 64, 512], [512, 10]
"""

from .BaseModel import BaseModel
import tensorflow as tf
import numpy as np
import scipy.stats as st


class LenetModel(BaseModel):

    # _________________________________________________________________________
    # build the neural network
    # create neural network with random initial parameters
    def _generate_random_parameters(self, parameters):
        layer_shapes = [[5, 5, 1, 32], [5, 5, 32, 64], [7 * 7 * 64, 512], [512, 10]]
        num_layers = 4

        initial_weights = [0] * num_layers
        initial_biases = [0] * num_layers
        # create initial parameters for the network
        for n in range(num_layers):
            initial_weights[n] = st.truncnorm(-2, 2, loc=0, scale=0.1).rvs(layer_shapes[n])
            initial_biases[n] = np.ones(layer_shapes[n][-1]) * 0.1

        return initial_weights, initial_biases

    # create a convolutional neural network with given initial parameters
    def _create_initialized_network(self, initial_weights, initial_biases):
        self._nn_weights = []
        self._nn_biases = []

        # create weights and biases of the neural network
        for init_w, init_b in zip(initial_weights, initial_biases):
            w = tf.Variable(init_w.astype(np.float32), dtype=tf.float32)
            b = tf.Variable(init_b.astype(np.float32), dtype=tf.float32)
            self._nn_weights += [w]
            self._nn_biases += [b]

        self._input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        self._target = tf.placeholder(tf.int32, shape=None)
        self._keep_prob = tf.placeholder(tf.float32)

        x = self._input
        # first convolutional layer
        y = tf.nn.conv2d(x, self._nn_weights[0], strides=[
                         1, 1, 1, 1], padding='SAME') + self._nn_biases[0]
        x = tf.nn.relu(y)

        # max pooling
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='SAME')

        # second convolutional layer
        y = tf.nn.conv2d(x, self._nn_weights[1], strides=[
                         1, 1, 1, 1], padding='SAME') + self._nn_biases[1]
        x = tf.nn.relu(y)

        # max pooling
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='SAME')

        # flatten the signal
        x = tf.reshape(x, [-1, initial_weights[2].shape[0]])

        # first fully connected layer, relu (for hidden layers)
        x = tf.nn.dropout(x, self._keep_prob)
        y = tf.matmul(x, self._nn_weights[2]) + self._nn_biases[2]
        z = tf.nn.relu(y)

        # output fully connected layer with softmax activation function
        x = tf.nn.dropout(z, self._keep_prob)
        y = tf.matmul(x, self._nn_weights[3]) + self._nn_biases[3]
        z = tf.nn.softmax(y)

        # output of the neural network
        self._logit = y
        self._output = z

        # loss function
        self._loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._target, logits=self._logit))

        # accuracy of the model
        matches = tf.equal(self._target, tf.argmax(self._logit, axis=1, output_type=tf.int32))
        self._accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

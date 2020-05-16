"""
   Convolutional neural network for classification of CIFAR10 data.
   The default is Lenet-5 like structure, two convolutional layers, followed by two fully connected ones.
   The filters' shapes are:
   [5, 5, 1, 32], [5, 5, 32, 64], [7 * 7 * 64, 384], [384, 192], [192, 10]
"""

from .BaseModel import BaseModel
import itertools
import numpy as np
import scipy.stats as st
import tensorflow as tf


class CifarNetModel(BaseModel):
    def __init__(self):
        super().__init__()

        self._image_size = 24

    # _________________________________________________________________________
    # build the neural network
    # create neural network with random initial parameters
    def _generate_random_parameters(self, parameters):
        flat_dim = self._image_size * self._image_size * 64 // 4 // 4
        layer_shapes = [[5, 5, 3, 64], [5, 5, 64, 64], [flat_dim, 384], [384, 192], [192, 10]]
        num_layers = len(layer_shapes)

        init_std = [0.05, 0.05, 0.04, 0.04, 1 / 192.0]
        init_bias = [0.0, 0.1, 0.1, 0.1, 0.0]
        initial_weights = [0] * num_layers
        initial_biases = [0] * num_layers

        # create initial parameters for the network
        for n in range(num_layers):
            initial_weights[n] = st.truncnorm(-2, 2, loc=0, scale=init_std[n]).rvs(layer_shapes[n])
            initial_biases[n] = np.ones(layer_shapes[n][-1]) * init_bias[n]

        return initial_weights, initial_biases

    # create a convolutional neural network with given initial parameters
    def _create_initialized_network(self, initial_weights, initial_biases):
        self._nn_weights = []
        self._nn_biases = []

        # create weights and biases of the neural network
        name_scopes = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
        for layer, init_w, init_b in zip(itertools.count(), initial_weights, initial_biases):
            with tf.variable_scope(name_scopes[layer]):
                w = tf.Variable(init_w.astype(np.float32), dtype=tf.float32, name='weights')
                b = tf.Variable(init_b.astype(np.float32), dtype=tf.float32, name='biases')

            self._nn_weights += [w]
            self._nn_biases += [b]

        self._input = tf.placeholder(tf.float32, shape=[None, self._image_size, self._image_size, 3])
        self._target = tf.placeholder(tf.int32, shape=None)
        self._keep_prob = tf.placeholder(tf.float32)

        x = self._input
        # convolutional layer 1
        y = tf.nn.conv2d(x, self._nn_weights[0], strides=[1, 1, 1, 1], padding='SAME') + self._nn_biases[0]
        x = tf.nn.relu(y, name=name_scopes[0])
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        # convolutional layer 2
        y = tf.nn.conv2d(x, self._nn_weights[1], strides=[1, 1, 1, 1], padding='SAME') + self._nn_biases[1]
        x = tf.nn.relu(y, name=name_scopes[1])
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # flatten the signal
        x = tf.reshape(x, [-1, initial_weights[2].shape[0]])

        # fully connected 1 (layer 3)
        x = tf.nn.dropout(x, self._keep_prob)
        y = tf.matmul(x, self._nn_weights[2]) + self._nn_biases[2]
        z = tf.nn.relu(y, name=name_scopes[2])

        # fully connected 2 (layer 4)
        x = tf.nn.dropout(z, self._keep_prob)
        y = tf.matmul(x, self._nn_weights[3]) + self._nn_biases[3]
        z = tf.nn.relu(y, name=name_scopes[3])

        # fully connected 3 (layer 5)
        x = tf.nn.dropout(z, self._keep_prob)
        y = tf.matmul(x, self._nn_weights[4]) + self._nn_biases[4]
        z = tf.nn.softmax(y, name=name_scopes[4])

        # output of the neural network
        self._logit = y
        self._output = z

        # loss function
        self._loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._target, logits=self._logit))

        # accuracy of the model
        matches = tf.equal(self._target, tf.argmax(self._logit, axis=1, output_type=tf.int32))
        self._accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

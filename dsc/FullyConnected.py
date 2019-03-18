"""
   Fully connected neural network for classification of data, hidden ReLU and final Softmax layers.
   The default network creates a 4 layers fully connected network, (784-300-100-10), for classification of MNSIT data.
"""

from .BaseModel import BaseModel
import tensorflow as tf
import numpy as np
import scipy.stats as st


class FCModel(BaseModel):

    # _________________________________________________________________________
    # build the neural network
    # create neural network with random initial parameters
    def _generate_random_parameters(self, parameters):
        layer_shapes = parameters.get('layer_shapes', [784, 300, 100, 10])
        num_layers = len(layer_shapes) - 1

        initial_weights = [0] * num_layers
        initial_biases = [0] * num_layers
        # create initial parameters for the network
        for n in range(num_layers):
            initial_weights[n] = st.truncnorm(-2, 2, loc=0, scale=0.1).rvs((layer_shapes[n], layer_shapes[n + 1]))
            initial_biases[n] = np.ones(layer_shapes[n + 1]) * 0.1

        return initial_weights, initial_biases

    # create a fully connected neural network with given initial parameters
    def _create_initialized_network(self, initial_weights, initial_biases):
        num_layers = len(initial_weights)
        input_len = initial_weights[0].shape[0]
        output_len = initial_weights[-1].shape[1]

        # create weights and biases of the neural network
        self._nn_weights = []
        self._nn_biases = []
        for init_w, init_b in zip(initial_weights, initial_biases):
            w = tf.Variable(init_w.astype(np.float32), dtype=tf.float32)
            b = tf.Variable(init_b.astype(np.float32), dtype=tf.float32)
            self._nn_weights += [w]
            self._nn_biases += [b]

        self._input = tf.placeholder(tf.float32, shape=[None, input_len])
        self._target = tf.placeholder(tf.int32, shape=[None, output_len])
        self._keep_prob = tf.placeholder(tf.float32)

        z = self._input
        for n in range(num_layers - 1):
            # create a fully connected layer with relu activation function
            x = tf.nn.dropout(z, self._keep_prob)
            y = tf.matmul(x, self._nn_weights[n]) + self._nn_biases[n]
            z = tf.nn.relu(y)

        # output layer of the neural network
        n = num_layers - 1
        x = tf.nn.dropout(z, self._keep_prob)
        y = tf.matmul(x, self._nn_weights[n]) + self._nn_biases[n]
        z = tf.nn.softmax(y)

        # outputs of the neural network
        self._logit = y
        self._output = z

        # loss function
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._target,
                                                                               logits=self._logit))

        # accuracy of the model
        matches = tf.equal(tf.argmax(self._target, 1), tf.argmax(self._logit, 1))
        self._accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

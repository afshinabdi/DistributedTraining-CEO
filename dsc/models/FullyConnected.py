"""
   Fully connected neural network for classification of data, hidden ReLU and final Softmax layers.
   The default network creates a 4 layers fully connected network, (784-300-100-10), for classification of MNSIT data.
"""

import tensorflow as tf
import numpy as np
import scipy.stats as st
from .BaseModel import BaseModel


class FCModel(BaseModel):
    # _________________________________________________________________________
    # build the neural network
    # create neural network with random initial parameters
    def _generate_random_parameters(self, settings):
        layer_shapes = settings.get('layer_shapes', [784, 300, 100, 10])
        self._num_layers = len(layer_shapes) - 1

        initial_params = [0] * self._num_layers
        # create initial parameters for the network
        for n in range(self._num_layers):
            w = st.truncnorm(-2, 2, loc=0, scale=0.1).rvs((layer_shapes[n], layer_shapes[n + 1]))
            b = np.ones(layer_shapes[n + 1]) * 0.1
            initial_params[n] = [w, b]

        return initial_params

    # create a fully connected neural network with given initial parameters
    def _create_initialized_network(self, initial_params, param_masks=None):
        if param_masks is None:
            param_masks = [None] * self._num_layers

        self._num_layers = len(initial_params)
        input_len = initial_params[0][0].shape[0]

        # create masks, weights and biases of the neural network
        self._nn_params = [None] * self._num_layers
        self._nn_masks = [None] * self._num_layers
        self._masked_params = [None] * self._num_layers
        for n in range(self._num_layers):
            w = tf.Variable(initial_params[n][0].astype(np.float32), dtype=tf.float32)
            b = tf.Variable(initial_params[n][1].astype(np.float32), dtype=tf.float32)
            self._nn_params[n] = [w, b]

            if param_masks[n] is None:
                mask = tf.constant(1, dtype=tf.float32)
            else:
                mask = tf.constant(param_masks[n], dtype=tf.float32)

            # mask is applied only to the weights, not biases
            self._masked_params[n] = [tf.multiply(w, mask), b]
            self._nn_masks[n] = mask

        self._input = tf.placeholder(tf.float32, shape=[None, input_len])
        self._target = tf.placeholder(tf.int32, shape=None)
        self._drop_rate = tf.placeholder(tf.float32)

        z = self._input
        for n in range(self._num_layers):
            # create a fully connected layer with relu activation function
            x = tf.nn.dropout(z, rate=self._drop_rate)
            y = tf.matmul(x, self._masked_params[n][0]) + self._masked_params[n][1]

            if n == self._num_layers - 1:
                # output layer of the neural network
                z = tf.nn.softmax(y)
            else:
                z = tf.nn.relu(y)

            self._fw_x += [x]
            self._fw_y += [y]


        # outputs of the neural network
        self._logit = y
        self._output = z

        # loss function
        self._loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._target, logits=self._logit))

        # accuracy of the model
        matches = tf.equal(self._target, tf.argmax(self._logit, axis=1, output_type=tf.int32))
        self._accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

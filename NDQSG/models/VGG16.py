########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
import scipy.stats as st
from scipy.misc import imread, imresize


class BasicVGGModel:
    def __init__(self):
        self._graph = None
        self._sess = None
        self._initializer = None
        self._accuracy = None

        self._trainOp = None
        self._learning_rate = 0.01
        self._loss = None
        self._gradients = None

        # parameters of the neural network
        self._keep_prob = None
        self._input = None
        self._output = None
        self._logit = None
        self._target = None

        self._nn_weights = []
        self._nn_biases = []
        self._fw_signals = []

    # =========================================================================
    # build the neural network
    def create_network(self, initial_weights=None, initial_biases=None):
        if initial_weights is None:
            initial_weights, initial_biases = self._create_random_network()

        self._create_initialized_network(initial_weights, initial_biases)

    def _add_convolutional_layer(self, x, kernel, bias, strides, padding):
        h = tf.Variable(kernel.astype(np.float32), dtype=tf.float32)
        b = tf.Variable(bias.astype(np.float32), dtype=tf.float32)

        self._fw_signals += [x]
        self._nn_weights += [h]
        self._nn_biases += [b]

        output = tf.nn.conv2d(x, h, strides=strides, padding=padding)
        output = tf.nn.relu(tf.nn.bias_add(output, b))

        return output

    def _add_fully_connected_layer(self, x, weight, bias, func=''):
        w = tf.Variable(weight.astype(np.float32), dtype=tf.float32)
        b = tf.Variable(bias.astype(np.float32), dtype=tf.float32)

        self._fw_signals += [x]
        self._nn_weights += [w]
        self._nn_biases += [b]

        output = tf.matmul(x, w) + b
        if func == 'relu':
            output = tf.nn.relu(output)
        elif func == 'softmax':
            self._logit = output
            output = tf.nn.softmax(output)

        return output

    def _add_max_pooling(self, x, ksize, stride, padding):
        output = tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)

        return output

    def _create_random_network(self):
        initial_weights = [0] * 16
        initial_biases = [0] * 16

        layer_shapes = [[3, 3, 3, 64], [3, 3, 64, 64], [3, 3, 64, 128], [3, 3, 128, 128], [3, 3, 128, 256],
                        [3, 3, 256, 256], [3, 3, 256, 256], [3, 3, 256, 512], [3, 3, 512, 512], [3, 3, 512, 512],
                        [3, 3, 512, 512], [3, 3, 512, 512], [3, 3, 512, 512], [25088, 4096], [4096, 4096], [4096, 1000]]

        for n in range(16):
            initial_weights[n] = st.truncnorm(-2, 2, loc=0, scale=0.1).rvs(layer_shapes[n])
            initial_biases[n] = np.ones(layer_shapes[n][-1]) * 0.1

        return initial_weights, initial_biases

    def _create_initialized_network(self, initial_weights, initial_biases):
        self._nn_weights = []
        self._nn_biases = []

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._input = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
            self._target = tf.placeholder(tf.float32, shape=[None, 1000])
            self._keep_prob = tf.placeholder(tf.float32)

            # pre-processing, zero-mean input
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3])
            x = self._input - mean

            self._fw_signals = []

            # create convolutional layers (when necessary, add a max-pooling)
            pooling_layers = [1, 3, 6, 9, 12]
            for n in range(13):
                x = self._add_convolutional_layer(x, initial_weights[n], initial_biases[n], strides=[1, 1, 1, 1],
                                                  padding='SAME')
                if n in pooling_layers:
                    x = self._add_max_pooling(x, 2, 2, padding='SAME')

            # create fully connected layers
            x = tf.reshape(x, [-1, initial_weights[13].shape[0]])
            for n in range(13, 16, 1):
                # drop-out
                x = tf.nn.dropout(x, keep_prob=self._keep_prob)
                if n < 15:
                    x = self._add_fully_connected_layer(x, initial_weights[n], initial_biases[n], func='relu')
                else:
                    x = self._add_fully_connected_layer(x, initial_weights[n], initial_biases[n], func='softmax')

            self._output = x
            self._fw_signals += [self._output]

            # loss function
            self._loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self._target, logits=self._logit))

            # accuracy of the model
            matches = tf.equal(tf.argmax(self._target, 1), tf.argmax(self._logit, 1))
            self._accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

    # =========================================================================
    # define optimizer of the neural network
    def create_optimizer(self, training_algorithm='Adam', learning_rate=0.01, decay_rate=0.95, decay_step=1000):
        with self._graph.as_default():
            # define the learning rate
            train_counter = tf.Variable(0, dtype=tf.float32)
            # decayed_learning_rate = learning_rate * decay_rate ^ (train_counter // decay_step)
            self._learning_rate = tf.train.exponential_decay(learning_rate, train_counter, decay_step,
                                                             decay_rate=decay_rate, staircase=True)

            # define the appropriate optimizer to use
            if (training_algorithm == 0) or (training_algorithm == 'GD'):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)
            elif (training_algorithm == 1) or (training_algorithm == 'RMSProp'):
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self._learning_rate)
            elif (training_algorithm == 2) or (training_algorithm == 'Adam'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
            elif (training_algorithm == 3) or (training_algorithm == 'AdaGrad'):
                optimizer = tf.train.AdagradOptimizer(learning_rate=self._learning_rate)
            elif (training_algorithm == 4) or (training_algorithm == 'AdaDelta'):
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._learning_rate)
            else:
                raise ValueError("Unknown training algorithm.")

            # =================================================================
            # training and initialization operators
            var_list = self._nn_weights + self._nn_biases
            self._trainOp = optimizer.minimize(self._loss, var_list=var_list, global_step=train_counter)

            gv = optimizer.compute_gradients(self._loss, var_list=var_list)
            self._gradients = [g for (g, v) in gv]

    # =========================================================================
    # create initializer and session to run the network
    def create_initializer(self):
        # initializer of the neural network
        with self._graph.as_default():
            self._initializer = tf.global_variables_initializer()

        self._sess = tf.Session(graph=self._graph)

    # =========================================================================
    # initialize the computation graph
    def initialize(self):
        if self._initializer is not None:
            self._sess.run(self._initializer)
        else:
            raise ValueError('Initializer has not been set.')

    # =========================================================================
    # compute the output of the network, (for top 5 accuracy, ... in codes)
    def compute_output(self, x):
        return self._sess.run(self._output, feed_dict={self._input: x, self._keep_prob: 1.0})

    # =========================================================================
    # One iteration of the training algorithm with input data
    def train(self, batch_x, batch_y, keep_prob=1):
        if self._trainOp is not None:
            self._sess.run(self._trainOp,
                           feed_dict={self._input: batch_x, self._target: batch_y, self._keep_prob: keep_prob})
        else:
            raise ValueError('Training algorithm has not been set.')

    def get_weights(self):
        return self._sess.run([self._nn_weights, self._nn_biases])

    def learning_rate(self):
        return self._sess.run(self._learning_rate)

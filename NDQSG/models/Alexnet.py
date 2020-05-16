"""
   Basic Alexnet neural network for image classification.
   This is a modified implementation of
      www.cs.toronto.edu/~guerzhoy/

    feed('data') -> conv(11, 11, 96, 4, 4, padding='VALID', name='conv1') -> lrn(2, 2e-05, 0.75, name='norm1') ->
    max_pool(3, 3, 2, 2, padding='VALID', name='pool1') -> conv(5, 5, 256, 1, 1, group=2, name='conv2') ->
    lrn(2, 2e-05, 0.75, name='norm2') -> max_pool(3, 3, 2, 2, padding='VALID', name='pool2') ->
    conv(3, 3, 384, 1, 1, name='conv3') -> conv(3, 3, 384, 1, 1, group=2, name='conv4') ->
    conv(3, 3, 256, 1, 1, group=2, name='conv5') -> max_pool(3, 3, 2, 2, padding='VALID', name='pool5') ->
    fc(4096, name='fc6') -> fc(4096, name='fc7') -> fc(1000, relu=False, name='fc8') -> softmax(name='prob')

"""

from .BaseModel import BaseModel
import tensorflow as tf
import numpy as np
import scipy.stats as st


class AlexnetModel(BaseModel):
    def __init__(self):
        super().__init__()

    # _________________________________________________________________________
    # build the neural network
    def _add_convolution_layer(self, x, kernel, bias, strides, padding):
        h = tf.Variable(kernel.astype(np.float32), dtype=tf.float32)
        b = tf.Variable(bias.astype(np.float32), dtype=tf.float32)

        self._nn_weights += [h]
        self._nn_biases += [b]

        output = tf.nn.conv2d(x, h, strides=strides, padding=padding)
        output = tf.nn.relu(tf.nn.bias_add(output, b))

        return output

    def _add_splitted_convolutional_layer(self, x, kernel, bias, strides, padding):
        h = tf.Variable(kernel.astype(np.float32), dtype=tf.float32)
        b = tf.Variable(bias.astype(np.float32), dtype=tf.float32)
        h0, h1 = tf.split(h, 2, axis=3)
        x0, x1 = tf.split(x, 2, axis=3)

        self._nn_weights += [h]
        self._nn_biases += [b]

        x0 = tf.nn.conv2d(x0, h0, strides=strides, padding=padding)
        x1 = tf.nn.conv2d(x1, h1, strides=strides, padding=padding)

        output = tf.concat([x0, x1], axis=3)
        output = tf.nn.relu(tf.nn.bias_add(output, b))

        return output

    def _add_fully_connected_layer(self, x, weight, bias, func=''):
        w = tf.Variable(weight.astype(np.float32), dtype=tf.float32)
        b = tf.Variable(bias.astype(np.float32), dtype=tf.float32)

        self._nn_weights += [w]
        self._nn_biases += [b]

        output = tf.matmul(x, w) + b
        if func == 'relu':
            output = tf.nn.relu(output)
        elif func == 'softmax':
            self._logit = output
            output = tf.nn.softmax(output)

        return output

    def _add_max_pooling(self, x):
        output = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        return output

    def _generate_random_parameters(self, parameters):
        layer_shapes = [[11, 11, 3, 96], [5, 5, 48, 256], [3, 3, 256, 384], [3, 3, 192, 384], [3, 3, 192, 256],
                        [9216, 4096], [4096, 4096], [4096, 1000]]

        num_layers = len(layer_shapes)

        initial_weights = [0] * num_layers
        initial_biases = [0] * num_layers
        # create initial parameters for the network
        for n in range(num_layers):
            initial_weights[n] = st.truncnorm(-2, 2, loc=0, scale=0.1).rvs(layer_shapes[n])
            initial_biases[n] = np.ones(layer_shapes[n][-1]) * 0.1

        return initial_weights, initial_biases

    def _create_initialized_network(self, initial_weights, initial_biases):
        input_dim = [None, 227, 227, 3]
        output_dim = [None, 1000]

        self._nn_weights = []
        self._nn_biases = []

        self._input = tf.placeholder(tf.float32, shape=input_dim)
        self._target = tf.placeholder(tf.float32, shape=output_dim)
        self._drop_rate = tf.placeholder(tf.float32)


        x = self._input

        # 1- convolution, local response normalization, and max-pooling
        x = self._add_convolution_layer(x, initial_weights[0], initial_biases[0], [1, 4, 4, 1], 'SAME')
        x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0)
        x = self._add_max_pooling(x)

        # 2- splitted convolution, local response normalization, and max-pooling
        x = self._add_splitted_convolutional_layer(x, initial_weights[1], initial_biases[1], [1, 1, 1, 1], 'SAME')
        x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0)
        x = self._add_max_pooling(x)

        # 3- only a convolution
        x = self._add_convolution_layer(x, initial_weights[2], initial_biases[2], [1, 1, 1, 1], 'SAME')

        # 4- splitted convolution
        x = self._add_splitted_convolutional_layer(x, initial_weights[3], initial_biases[3], [1, 1, 1, 1], 'SAME')

        # 5- splitted convolutional layer, max-pooling
        x = self._add_splitted_convolutional_layer(x, initial_weights[4], initial_biases[4], [1, 1, 1, 1], 'SAME')
        x = self._add_max_pooling(x)

        # 6- fully connected layer, relu
        x = tf.reshape(x, [-1, initial_weights[5].shape[0]])
        x = tf.nn.dropout(x, rate=self._drop_rate)
        x = self._add_fully_connected_layer(x, initial_weights[5], initial_biases[5], func='relu')

        # 7- another fully connected layer, relu
        x = tf.nn.dropout(x, rate=self._drop_rate)
        x = self._add_fully_connected_layer(x, initial_weights[6], initial_biases[6], func='relu')

        # 8- output fully connected layer, softmax
        x = tf.nn.dropout(x, rate=self._drop_rate)
        self._output = self._add_fully_connected_layer(x, initial_weights[7], initial_biases[7], func='softmax')

        # loss function
        self._loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._target, logits=self._logit))

        # =================================================================
        # accuracy of the model
        matches = tf.equal(tf.argmax(self._target, 1), tf.argmax(self._logit, 1))
        self._accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

        # computing accuracy
        matches = tf.nn.in_top_k(predictions=self._output, targets=tf.argmax(self._target, 1), k=5)
        self._accuracy_top5 = tf.reduce_mean(tf.cast(matches, tf.float32))

        matches = tf.equal(tf.argmax(self._target, 1), tf.argmax(self._logit, 1))
        self._accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))


    # _________________________________________________________________________
    # compute the accuracy of the NN using the given inputs
    def compute_accuracy(self, x, target, top_5=False):
        if top_5:
            return self._sess.run([self._accuracy, self._accuracy_top5],
                                  feed_dict={self._input: x, self._target: target, self._drop_rate: 0})
        else:
            return self._sess.run(self._accuracy,
                                  feed_dict={self._input: x, self._target: target, self._drop_rate: 0})


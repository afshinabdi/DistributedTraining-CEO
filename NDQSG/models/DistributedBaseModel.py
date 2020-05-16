"""
    Base class to simulate a network of multiple workers.
    input parameter to create the neural network may have the following fields:
        initial_w, initial_b: initial weights and biases of the neural network, 
                              if not provided the child class will generate them randomly based on its sructure
        l1_regularizer:
        l2_regularizer:
        training_alg:
        learning_rate:
        decay_rate:
        decay_step:
        compute_gradients:
        assign_operator:
"""

import tensorflow as tf
import distributed_data.sg_quantizers.tf_implementation.cs_quantizer as tf_csq
import distributed_data.sg_quantizers.tf_implementation.qsg_quantizer as tf_qsg
import distributed_data.sg_quantizers.tf_implementation.dithered_quantizer as tf_dq
import distributed_data.sg_quantizers.tf_implementation.dithered_transform_quantizer as tf_dtq
import distributed_data.sg_quantizers.tf_implementation.onebit_quantizer as tf_obq
import distributed_data.sg_quantizers.tf_implementation.topK_sgd as tf_topK

_DEFAULT_SEED = 94635


class DistributedBaseModel:
    def __init__(self):
        # parameters of the workers
        self._number_workers = 0

        # parameters of the neural network
        self._sess = None
        self._initializer = None
        self._accuracy = None

        self._optimizer = None
        self._trainOp = None
        self._global_step = None
        self._learning_rate = 0.01
        self._loss = None

        # input, output of the model
        self._drop_rate = None
        self._input = None
        self._output = None
        self._logit = None
        self._target = None

        # parameters of the neural network
        self._num_layers = 0
        self._nn_weights = []
        self._nn_biases = []

        # gradients
        self._gW = None
        self._gb = None

        # reconstructed quantized gradients
        self._gWh = None
        self._gbh = None

        # to apply externally computed gradients
        self._input_gW = None
        self._input_gb = None
        self._apply_gradients = None

    # _________________________________________________________________________
    # build the neural network
    def create_network(self, parameters: dict):
        self._number_workers = parameters.get('num workers', 1)    # number of workers
        seed = parameters.get('seed', _DEFAULT_SEED)    # graph level seed

        if parameters.get('initial_w') is None:
            initial_weights, initial_biases = self._generate_random_parameters(parameters)
        else:
            initial_weights = parameters.get('initial_w')
            initial_biases = parameters.get('initial_b')

        self._num_layers = len(initial_weights)

        graph = tf.Graph()
        with graph.as_default():
            # set graph level random number seed
            tf.set_random_seed(seed)

            # 1- create the neural network with the given/random initial weights/biases
            self._create_initialized_network(initial_weights, initial_biases)

            # 2- if required, add regularizer to the loss function
            l1 = parameters.get('l1_regularizer')
            if l1 is not None:
                self._add_l1regulizer(w=l1)

            l2 = parameters.get('l2_regularizer')
            if l2 is not None:
                self._add_l2regulizer(w=l2)

            # 3- if requried, add the training algorithm
            alg = parameters.get('training_alg')
            if alg is not None:
                self._add_optimizer(parameters)

                # 4- compute gradients? only if optimizer is defined
                if parameters.get('compute_gradients', False):
                    self._add_gradient_computations(parameters)

            initializer = tf.global_variables_initializer()

        self._sess = tf.Session(graph=graph)
        self._sess.run(initializer)

    # _________________________________________________________________________
    # create neural network with random initial parameters
    def _generate_random_parameters(self, parameters):
        pass

    # create a fully connected neural network with given initial parameters
    def _create_initialized_network(self, initial_weights, initial_biases):
        pass

    # _________________________________________________________________________
    # add regulizer to the loss function
    def _add_l1regulizer(self, w):
        if type(w) is float:
            w = [w] * self._num_layers

        assert len(w) == self._num_layers, 'Not enough weights for the regularizer.'

        l1_loss = tf.add_n([(s * tf.norm(v, ord=1)) for (v, s) in zip(self._nn_weights, w)])
        self._loss += l1_loss

    def _add_l2regulizer(self, w):
        if type(w) is float:
            w = [w] * self._num_layers

        assert len(w) == self._num_layers, 'Not enough weights for the regularizer.'

        l2_loss = tf.add_n([(s * tf.nn.l2_loss(v)) for (v, s) in zip(self._nn_weights, w)])
        self._loss += l2_loss

    # _________________________________________________________________________
    # define optimizer of the neural network
    def _add_optimizer(self, parameters):
        alg = parameters.get('training_alg', 'GD')
        lr = parameters.get('learning_rate', 0.01)
        dr = parameters.get('decay_rate', 0.95)
        ds = parameters.get('decay_step', 200)

        # define the learning rate
        self._global_step = tf.Variable(0, dtype=tf.float32)
        # decayed_learning_rate = learning_rate * dr ^ (global_step // ds)
        self._learning_rate = tf.train.exponential_decay(lr, self._global_step, ds, decay_rate=dr, staircase=True)

        # define the appropriate optimizer to use
        if (alg == 0) or (alg == 'GD'):
            self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)
        elif (alg == 1) or (alg == 'RMSProp'):
            self._optimizer = tf.train.RMSPropOptimizer(learning_rate=self._learning_rate)
        elif (alg == 2) or (alg == 'Adam'):
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        elif (alg == 3) or (alg == 'AdaGrad'):
            self._optimizer = tf.train.AdagradOptimizer(learning_rate=self._learning_rate)
        elif (alg == 4) or (alg == 'AdaDelta'):
            self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._learning_rate)
        else:
            raise ValueError("Unknown training algorithm.")

        # =================================================================
        # training and initialization operators
        var_list = self._nn_weights + self._nn_biases
        self._trainOp = self._optimizer.minimize(self._loss, var_list=var_list, global_step=self._global_step)

    def _add_gradient_computations(self, parameters):
        # computing gradients
        self._gW = tf.gradients(self._loss, self._nn_weights)
        self._gb = tf.gradients(self._loss, self._nn_biases)

        # applying gradients to the optimizer
        self._input_gW = tuple([tf.placeholder(dtype=tf.float32, shape=w.get_shape()) for w in self._nn_weights])
        self._input_gb = tuple([tf.placeholder(dtype=tf.float32, shape=b.get_shape()) for b in self._nn_biases])
        gv = [(g, v) for g, v in zip(self._input_gW, self._nn_weights)]
        gv += [(g, v) for g, v in zip(self._input_gb, self._nn_biases)]

        self._apply_gradients = self._optimizer.apply_gradients(gv, global_step=self._global_step)

        if parameters.get('quantizer') is not None:
            self._add_gradient_quantizers(parameters)

    # _________________________________________________________________________
    # add the computations for the gradient quantizer
    def _add_gradient_quantizers(self, parameters):
        quantization_method = parameters.get('quantizer', '')

        # random number generation seed
        seeds = parameters.get('quantizer_seeds')

        self._gWh = [[0] * self._num_layers for _ in range(self._number_workers)]
        self._gbh = [[0] * self._num_layers for _ in range(self._number_workers)]

        # operators to update the quantization reside (if necessary)
        self._updateRw = [[tf.no_op()] * self._num_layers for _ in range(self._number_workers)]
        self._updateRb = [[tf.no_op()] * self._num_layers for _ in range(self._number_workers)]

        if quantization_method == '':
            for nw in range(self._number_workers):
                self._gWh[nw], self._gbh[nw] = self._gW, self._gb

        # add operations for quantization and reconstruction of gradients
        elif quantization_method == 'one-bit':
            for nw in range(self._number_workers):
                for layer in range(self._num_layers):
                    _, _, _, gwh, urw = tf_obq.quantize(self._gW[layer])
                    _, _, _, gbh, urb = tf_obq.quantize(self._gb[layer])

                    self._gWh[nw][layer], self._gbh[nw][layer] = gwh, gbh
                    self._updateRw[nw][layer], self._updateRb[nw][layer] = urw, urb

        elif quantization_method == 'dithered':
            bucket_sizes = parameters.get('bucket_sizes')
            num_levels = parameters.get('num_levels')

            for nw in range(self._number_workers):
                for layer in range(self._num_layers):
                    _, _, gwh = tf_dq.quantize(
                        self._gW[layer], num_levels, bucket_sizes[layer][0], seeds[nw][2 * layer]
                    )
                    _, _, gbh = tf_dq.quantize(
                        self._gb[layer], num_levels, bucket_sizes[layer][1], seeds[nw][2 * layer + 1]
                    )

                    self._gWh[nw][layer], self._gbh[nw][layer] = gwh, gbh

        elif quantization_method == 'dithered-transform':
            bucket_sizes = parameters.get('bucket_sizes')
            num_levels = parameters.get('num_levels')
            H_matrices = parameters.get('H')

            for nw in range(self._number_workers):
                for layer in range(self._num_layers):
                    _, _, gwh = tf_dtq.quantize(
                        self._gW[layer], H_matrices[layer][0], num_levels, bucket_sizes[layer][0], seeds[nw][2 * layer]
                    )
                    _, _, gbh = tf_dtq.quantize(
                        self._gb[layer], H_matrices[layer][1], num_levels, bucket_sizes[layer][1],
                        seeds[nw][2 * layer + 1]
                    )

                    self._gWh[nw][layer], self._gbh[nw][layer] = gwh, gbh

        elif quantization_method == 'qsg':
            bucket_sizes = parameters.get('bucket_sizes')
            num_levels = parameters.get('num_levels')

            for nw in range(self._number_workers):
                for layer in range(self._num_layers):
                    _, _, gwh = tf_qsg.quantize(self._gW[layer], num_levels, bucket_sizes[layer][0])
                    _, _, gbh = tf_qsg.quantize(self._gb[layer], num_levels, bucket_sizes[layer][1])

                    self._gWh[nw][layer], self._gbh[nw][layer] = gwh, gbh

        elif quantization_method == 'quantized-cs':
            num_levels = parameters.get('num_levels')
            T_matrices = parameters.get('H')
            use_Rademacher = parameters.get('use_Rademacher', False)
            shuffle_T = parameters.get('shuffle_T', True)
            err_feedback = parameters.get('error_feedback', True)

            for nw in range(self._number_workers):
                for layer in range(self._num_layers):
                    _, _, gwh, urw, _ = tf_csq.quantize(
                        self._gW[layer], T_matrices[layer][0], num_levels, seeds[nw][2 * layer], use_Rademacher,
                        shuffle_T, err_feedback
                    )
                    _, _, gbh, urb, _ = tf_csq.quantize(
                        self._gb[layer], T_matrices[layer][1], num_levels, seeds[nw][2 * layer + 1], use_Rademacher,
                        shuffle_T, err_feedback
                    )

                    self._gWh[nw][layer], self._gbh[nw][layer] = gwh, gbh

    # _________________________________________________________________________
    # compute the accuracy of the NN using the given inputs
    def accuracy(self, x, y):
        return self._sess.run(self._accuracy, feed_dict={self._input: x, self._target: y, self._drop_rate: 0.})

    # _________________________________________________________________________
    # compute the output of the NN to the given inputs
    def output(self, x):
        return self._sess.run(self._output, feed_dict={self._input: x, self._drop_rate: 0.})

    # _________________________________________________________________________
    # One iteration of the training algorithm with input data
    def train(self, x, y, drop_rate=0):
        assert self._trainOp is not None, 'Training algorithm has not been set.'

        self._sess.run(self._trainOp, feed_dict={self._input: x, self._target: y, self._drop_rate: drop_rate})

    def get_weights(self):
        return self._sess.run([self._nn_weights, self._nn_biases])

    def learning_rate(self):
        return self._sess.run(self._learning_rate)

    # _________________________________________________________________________
    # Compute the gradients of the parameters of the NN for the given input
    def get_gradients(self, x, y, drop_rate=0.):
        assert self._gW is not None, 'The operators to compute the gradients have not been defined.'

        return self._sess.run(
            [self._gW, self._gb], feed_dict={
                self._input: x,
                self._target: y,
                self._drop_rate: drop_rate
            }
        )

    # _________________________________________________________________________
    # Quantize the gradients of the parameters of the NN for the given input
    def quantized_gradients(self, x, y, drop_rate=0., worker_idx=0):
        assert self._gWh is not None, 'The operators to quantize the gradients have not been defined.'

        return self._sess.run(
            [self._gWh[worker_idx], self._gbh[worker_idx], self._updateRw[worker_idx], self._updateRb[worker_idx]],
            feed_dict={
                self._input: x,
                self._target: y,
                self._drop_rate: drop_rate
            }
        )

    # _________________________________________________________________________
    # Apply the gradients externally computed to the optimizer
    def apply_gradients(self, gw, gb):
        assert self._apply_gradients is not None, 'The operators to apply the gradients have not been defined.'

        feed_dict = {self._input_gW: gw, self._input_gb: gb}
        self._sess.run(self._apply_gradients, feed_dict=feed_dict)

    # _________________________________________________________________________
    @property
    def number_layers(self):
        return self._num_layers
"""
    Base model to define other neural networks.
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


class BaseModel:
    def __init__(self):
        self._sess = None
        self._initializer = None
        self._accuracy = None

        self._optimizer = None
        self._trainOp = None
        self._global_step = None
        self._learning_rate = 0.01
        self._loss = None

        # parameters of the neural network
        self._keep_prob = None
        self._input = None
        self._output = None
        self._logit = None
        self._target = None

        self._nn_weights = []
        self._nn_biases = []

        # computing gradients
        self._grad_W = None
        self._grad_b = None

        # to apply externally computed gradients
        self._input_gW = None
        self._input_gb = None
        self._apply_gradients = None

        # variables to set parameters during training or after model being initialized
        self._assign_op = None
        self._input_weights = None
        self._input_biases = None

    # _________________________________________________________________________
    # build the neural network
    def create_network(self, parameters: dict):

        if parameters.get('initial_w') is None:
            initial_weights, initial_biases = self._generate_random_parameters(parameters)
        else:
            initial_weights = parameters.get('initial_w')
            initial_biases = parameters.get('initial_b')

        graph = tf.Graph()
        with graph.as_default():
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
                    self._add_gradient_computations()

            # 5- operators to assign weights/biases?
            if parameters.get('assign_operator', False):
                self._add_assign_operators()

            self._initializer = tf.global_variables_initializer()

        self._sess = tf.Session(graph=graph)

    # _________________________________________________________________________
    # create neural network with random initial parameters
    def _generate_random_parameters(self, parameters):
        pass

    # create a fully connected neural network with given initial parameters
    def _create_initialized_network(self, initial_weights, initial_biases):
        pass

    # _________________________________________________________________________
    # update (assign) operator for the parameters of the NN model
    def _add_assign_operators(self):
        self._assign_op = []
        self._input_weights = ()
        self._input_biases = ()

        for w in self._nn_weights:
            w_placeholder = tf.placeholder(dtype=tf.float32, shape=w.get_shape())
            w_assign_op = w.assign(w_placeholder)
            self._assign_op.append(w_assign_op)
            self._input_weights += (w_placeholder,)

        for b in self._nn_biases:
            b_placeholder = tf.placeholder(dtype=tf.float32, shape=b.get_shape())
            b_assign_op = b.assign(b_placeholder)
            self._assign_op.append(b_assign_op)
            self._input_biases += (b_placeholder,)

    # _________________________________________________________________________
    # add regulizer to the loss function
    def _add_l1regulizer(self, w):
        num_layers = len(self._nn_weights)

        if type(w) is float:
            w = [w] * num_layers

        assert len(w) == num_layers, 'Not enough weights for the regularizer.'

        l1_loss = tf.add_n([(s * tf.norm(v, ord=1))
                            for (v, s) in zip(self._nn_weights, w)])
        self._loss += l1_loss

    def _add_l2regulizer(self, w):
        num_layers = len(self._nn_weights)

        if type(w) is float:
            w = [w] * num_layers

        assert len(w) == num_layers, 'Not enough weights for the regularizer.'

        l2_loss = tf.add_n([(s * tf.nn.l2_loss(v))
                            for (v, s) in zip(self._nn_weights, w)])
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
        # decayed_learning_rate = learning_rate * dr ^ (gloval_step // ds)
        self._learning_rate = tf.train.exponential_decay(lr, self._global_step, ds, decay_rate=dr, staircase=True)

        # define the appropriate optimizer to use
        if (alg == 0) or (alg == 'GD'):
            self._optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self._learning_rate)
        elif (alg == 1) or (alg == 'RMSProp'):
            self._optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self._learning_rate)
        elif (alg == 2) or (alg == 'Adam'):
            self._optimizer = tf.train.AdamOptimizer(
                learning_rate=self._learning_rate)
        elif (alg == 3) or (alg == 'AdaGrad'):
            self._optimizer = tf.train.AdagradOptimizer(
                learning_rate=self._learning_rate)
        elif (alg == 4) or (alg == 'AdaDelta'):
            self._optimizer = tf.train.AdadeltaOptimizer(
                learning_rate=self._learning_rate)
        else:
            raise ValueError("Unknown training algorithm.")

        # =================================================================
        # training and initialization operators
        var_list = self._nn_weights + self._nn_biases
        self._trainOp = self._optimizer.minimize(self._loss, var_list=var_list, global_step=self._global_step)

    def _add_gradient_computations(self):
        # computing gradients
        self._grad_W = tf.gradients(self._loss, self._nn_weights)
        self._grad_b = tf.gradients(self._loss, self._nn_biases)

        # applying gradients to the optimizer
        self._input_gW = tuple([tf.placeholder(dtype=tf.float32, shape=w.get_shape()) for w in self._nn_weights])
        self._input_gb = tuple([tf.placeholder(dtype=tf.float32, shape=b.get_shape()) for b in self._nn_biases])
        gv = [(g, v) for g, v in zip(self._input_gW, self._nn_weights)]
        gv += [(g, v) for g, v in zip(self._input_gb, self._nn_biases)]

        self._apply_gradients = self._optimizer.apply_gradients(gv, global_step=self._global_step)

    # _________________________________________________________________________
    # initialize the computation graph
    def initialize(self):
        assert self._sess is not None, 'The model has not been created.'

        self._sess.run(self._initializer)

    # _________________________________________________________________________
    # compute the accuracy of the NN using the given inputs
    def accuracy(self, x, y):
        return self._sess.run(self._accuracy, feed_dict={self._input: x, self._target: y, self._keep_prob: 1.0})

    # compute the output of the NN to the given inputs

    def output(self, x):
        return self._sess.run(self._output, feed_dict={self._input: x, self._keep_prob: 1.0})

    # _________________________________________________________________________
    # One iteration of the training algorithm with input data
    def train(self, x, y, keep_prob=1):
        assert self._trainOp is not None, 'Training algorithm has not been set.'

        self._sess.run(self._trainOp, feed_dict={self._input: x, self._target: y, self._keep_prob: keep_prob})

    def get_weights(self):
        return self._sess.run([self._nn_weights, self._nn_biases])

    def set_weights(self, new_weights, new_biases):
        assert self._assign_op is not None, 'The assign operators has been added to the graph.'

        self._sess.run(self._assign_op, feed_dict={self._input_weights: new_weights, self._input_biases: new_biases})

    def learning_rate(self):
        return self._sess.run(self._learning_rate)

    # _________________________________________________________________________
    # Compute the gradients of the parameters of the NN for the given input
    def compute_gradients(self, x, y):
        assert self._grad_W is not None, 'The operators to compute the gradients have not been defined.'

        return self._sess.run([self._grad_W, self._grad_b], feed_dict={self._input: x, self._target: y,
                                                                       self._keep_prob: 1.0})

    # Apply the gradients externally computed to the optimizer

    def apply_gradients(self, gw, gb):
        assert self._apply_gradients is not None, 'The operators to apply the gradients have not been defined.'

        feed_dict = {self._input_gW: gw, self._input_gb: gb}
        self._sess.run(self._apply_gradients, feed_dict=feed_dict)

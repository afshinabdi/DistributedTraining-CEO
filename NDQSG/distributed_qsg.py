"""
    Simulation of 'QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding'
"""

import quantizers.qsg_quantizer as quantizer


class WorkerNode:
    def __init__(self, nn):
        self._nn = nn

        # quantization parameters
        self._bucket_size = None
        self._num_levels = 1

    # =========================================================================
    # set parameters of the quantizer
    def set_quantizer(self, bucket_size=None, num_levels=1):
        self._bucket_size = bucket_size
        self._num_levels = num_levels

    # =========================================================================
    # compute the gradients and 1-bit quantize them for transmission
    def get_quantized_gradients(self, x, y):
        gW, gb = self._nn.compute_gradients(x, y)

        if self._bucket_size is None:
            self._bucket_size = [[None] * len(gW), [None] * len(gb)]

        # quantize the gradients of the weights
        d = self._bucket_size[0]
        q_gW = [0] * len(gW)
        s_gW = [0] * len(gW)

        for n in range(len(gW)):
            q_gW[n], s_gW[n] = quantizer.quantize(gW[n], d=d[n], num_levels=self._num_levels)  # quantize

        # quantize the gradients of the biases
        d = self._bucket_size[1]

        q_gb = [0] * len(gb)
        s_gb = [0] * len(gb)

        for n in range(len(gb)):
            q_gb[n], s_gb[n] = quantizer.quantize(gb[n], d=d[n], num_levels=self._num_levels)  # quantize

        return q_gW, s_gW, q_gb, s_gb

    # =========================================================================
    # apply the received gradient (from server, ...) to the neural network
    def apply_gradients(self, gW, gb):
        self._nn.apply_gradients(gW, gb)


class AggregationNode:
    def __init__(self):
        self._gW = None
        self._gb = None
        self._num_workers = 0

        # parameters of each worker's quantizer
        self._bucket_size = None
        self._num_levels = 1

    # =========================================================================
    # set parameters of the quantizer of each worker
    def set_quantizer(self, bucket_size=None, num_levels=1):
        self._bucket_size = bucket_size
        self._num_levels = num_levels

    # =========================================================================
    # reset the parameters of the algorithm
    def reset_node(self):
        self._num_workers = 0
        self._gW = None
        self._gb = None

    # =========================================================================
    # get computed and quantized SG from a worker and aggregate the gradient
    def receive_gradient(self, qW, sW, qb, sb):
        self._num_workers += 1
        if self._bucket_size is None:
            self._bucket_size = [[None] * len(qW), [None] * len(qb)]

        # gradients of the weights
        if self._gW is None:
            self._gW = [0] * len(qW)

        d = self._bucket_size[0]

        for n in range(len(qW)):
            g = quantizer.dequantize(qW[n], sW[n], d[n])
            self._gW[n] += g

        # gradients of the biases
        if self._gb is None:
            self._gb = [0] * len(qb)

        d = self._bucket_size[1]

        for n in range(len(qb)):
            g = quantizer.dequantize(qb[n], sb[n], d[n])
            self._gb[n] += g

    # =========================================================================
    # get computed and quantized SG from a worker and aggregate the gradient
    def get_aggregated_gradients(self):
        if self._num_workers == 0:
            return 0, 0

        for n in range(len(self._gW)):
            self._gW[n] /= self._num_workers

        for n in range(len(self._gb)):
            self._gb[n] /= self._num_workers

        return self._gW, self._gb


"""
    # A simple code that uses only one neural network model for computations is as follows:

    num_bits = 2
    bucket_size = [[128, 128, ...], [None, None, ...]]
    nn = GradientModel()
    ...
    worker = [WorkerNode(nn) for _ in range(num_workers)]
    server = AggregationNode()
    ...
    server.reset_node()
    for w in range(num_workers):
        qw, sw, qb, sb = worker[w].get_quantized_gradients(x[w], y[w])
        server.receive_gradient(qw, sw, qb, sb)

    gw, gb = server.get_aggregated_gradients()
    nn.apply_gradients(gw, gb)
"""

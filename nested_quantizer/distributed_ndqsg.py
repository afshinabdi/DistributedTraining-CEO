"""
    Simulation of 'nested dithered quantized stochastic gradient'
"""

import numpy as np
import dithered_quantizer as quantizer

default_seed = 798946
min_seed = 1000
max_seed = 1000000


class WorkerNode:
    def __init__(self, nn):
        self._nn = nn

        # quantization parameters
        self._seed = default_seed
        self._clip_thr = None
        self._bucket_size = None
        self._num_levels = 1
        self._alpha = 1.0
        self._nested = False

    # =========================================================================
    # set parameters of the quantizer
    def set_quantizer(self, seed, clip_thr=None, bucket_size=None, num_levels=1, alpha=1.0):
        self._seed = seed
        self._clip_thr = clip_thr
        self._bucket_size = bucket_size
        self._num_levels = num_levels
        self._alpha = alpha

        if type(self._num_levels) is int:
            self._nested = False
        else:
            if len(self._num_levels) != 2:
                raise ValueError('For nested quantization, 2 quantizers should be given.')

            self._nested = True

    # =========================================================================
    # quantize the computed gradients
    def get_quantized_gradients(self, x, y):
        gW, gb = self._nn.compute_gradients(x, y)
        np.random.seed(self._seed)

        if self._bucket_size is None:
            self._bucket_size = [[None] * len(gW), [None] * len(gb)]

        # quantize the gradients of the weights
        d = self._bucket_size[0]
        q_gW = [0] * len(gW)
        s_gW = [0] * len(gW)
        for n, g in enumerate(gW):
            # 1- clip the gradients
            if self._clip_thr is not None:
                thr = self._clip_thr * np.std(g)
                g = np.clip(g, -thr, thr)

            # 2- quantize gradients
            if self._nested:
                q_gW[n], s_gW[n] = quantizer.nested_quantize(g, self._num_levels, d[n], self._alpha)
            else:
                q_gW[n], s_gW[n] = quantizer.quantize(g, num_levels=self._num_levels, d=d[n])

        # quantize the gradients of the biases
        d = self._bucket_size[1]
        q_gb = [0] * len(gb)
        s_gb = [0] * len(gb)
        for n, g in enumerate(gb):
            # 1- clip the gradients
            if self._clip_thr is not None:
                thr = self._clip_thr * np.std(g)
                g = np.clip(g, -thr, thr)

            # 2- quantize gradients
            if self._nested:
                q_gb[n], s_gb[n] = quantizer.nested_quantize(g, self._num_levels, d[n], self._alpha)
            else:
                q_gb[n], s_gb[n] = quantizer.quantize(g, num_levels=self._num_levels, d=d[n])

        # update the seed
        self._seed = np.random.randint(min_seed, max_seed)

        return q_gW, s_gW, q_gb, s_gb

    # =========================================================================
    # apply the received gradient (from server, ...) to the neural network
    def apply_gradients(self, gW, gb):
        self._nn.apply_gradients(gW, gb)


class AggregationNode:
    def __init__(self, num_workers):
        self._gW = None
        self._gb = None
        self._num_received = 0
        self._num_workers = num_workers

        # parameters of each worker's quantizer
        self._seeds = [default_seed] * self._num_workers
        self._bucket_size = [None] * self._num_workers
        self._num_levels = [1] * self._num_workers
        self._alpha = [1.0] * self._num_workers
        self._nested = [False] * self._num_workers

    # =========================================================================
    # set parameters of the quantizer of each worker
    def set_quantizer(self, worker_id, seed=default_seed, bucket_size=None, num_levels=1, alpha=1.0):
        if worker_id >= self._num_workers:
            raise ValueError('Id if the input worker should be smaller than the number of workers.')

        self._seeds[worker_id] = seed
        self._bucket_size[worker_id] = bucket_size
        self._num_levels[worker_id] = num_levels
        self._alpha[worker_id] = alpha

        if type(num_levels) is int:
            self._nested[worker_id] = False
        else:
            if len(num_levels) != 2:
                raise ValueError('For nested quantization, 2 quantizers should be given.')

            self._nested[worker_id] = True

    # =========================================================================
    # reset the parameters of the algorithm
    def reset_node(self):
        self._num_received = 0
        self._gW = None
        self._gb = None

    # =========================================================================
    # receive gradients from a worker
    def receive_gradient(self, worker_id, qW, sW, qb, sb):
        np.random.seed(self._seeds[worker_id])

        self._num_received += 1
        gain = 1.0 / self._num_received

        alpha = self._alpha[worker_id]
        num_levels = self._num_levels[worker_id]
        if self._bucket_size[worker_id] is None:
            self._bucket_size[worker_id] = [[None] * len(qW), [None] * len(qb)]

        # gradients of the weights
        if self._gW is None:
            self._gW = [0] * len(qW)

        d = self._bucket_size[worker_id][0]
        for n in range(len(qW)):
            if self._nested[worker_id]:
                g = quantizer.nested_dequantize(qW[n], sW[n], d[n], alpha, y=self._gW[n], num_levels=num_levels)
            else:
                g = quantizer.dequantize(qW[n], sW[n], d[n])

            self._gW[n] = gain * g + (1 - gain) * self._gW[n]

        # gradients of the biases
        if self._gb is None:
            self._gb = [0] * len(qb)

        d = self._bucket_size[worker_id][1]
        for n in range(len(qb)):
            if self._nested[worker_id]:
                g = quantizer.nested_dequantize(qb[n], sb[n], d[n], alpha, y=self._gb[n], num_levels=num_levels)
            else:
                g = quantizer.dequantize(qb[n], sb[n], d[n])

            self._gb[n] = gain * g + (1 - gain) * self._gb[n]

        # update the seed
        self._seeds[worker_id] = np.random.randint(min_seed, max_seed)

    # =========================================================================
    # get computed and quantized SG from a worker and aggregate the gradient
    def get_aggregated_gradients(self):
        if self._num_received == 0:
            return 0, 0

        if self._num_workers == self._num_received:
            return self._gW, self._gb
        else:
            # only part of data has been received, scale accordingly
            scale = float(self._num_received) / float(self._num_workers)
            for n in range(len(self._gW)):
                self._gW[n] *= scale

            for n in range(len(self._gb)):
                self._gb[n] *= scale

            return self._gW, self._gb

"""
    Simulation of 1-bit stochastic gradient method for data parallel distributed training
"""

import itertools
import onebit_quantizer as quantizer


class WorkerNode:
    def __init__(self, nn):
        self._nn = nn
        self._residue_gW = None
        self._residue_gb = None

    # =========================================================================
    # compute the gradients and 1-bit quantize them for transmission
    def get_quantized_gradients(self, x, y):
        gW, gb = self._nn.compute_gradients(x, y)

        if self._residue_gW is None:
            self._residue_gW = [0] * len(gW)

        if self._residue_gb is None:
            self._residue_gb = [0] * len(gb)

        # quantize the gradients of the weights
        q_gW = [0] * len(gW)
        c_gW = [0] * len(gW)

        for n, g in enumerate(gW):
            g = g + self._residue_gW[n]  # add residue
            q_gW[n], c_gW[n], self._residue_gW[n] = quantizer.quantize(g)  # quantize

        # quantize the gradients of the weights
        q_gb = [0] * len(gb)
        c_gb = [0] * len(gb)

        for n, g, r in zip(itertools.count(), gb, self._residue_gb):
            g = g + r  # add residue
            q_gb[n], c_gb[n], self._residue_gb[n] = quantizer.quantize(g)  # quantize

        return q_gW, c_gW, q_gb, c_gb

    # =========================================================================
    # apply the received gradient (from server, ...) to the neural network
    def apply_gradients(self, gW, gb):
        self._nn.apply_gradients(gW, gb)


class AggregationNode:
    def __init__(self):
        self._gW = None
        self._gb = None
        self._num_workers = 0

    # =========================================================================
    # reset the parameters of the algorithm
    def reset_node(self):
        self._num_workers = 0
        self._gW = None
        self._gb = None

    # =========================================================================
    # get computed and quantized SG from a worker and aggregate the gradient
    def receive_gradient(self, qW, cW, qb, cb):
        self._num_workers += 1

        # gradients of the weights
        if self._gW is None:
            self._gW = [0] * len(qW)

        for n, q, c in zip(itertools.count(), qW, cW):
            g = quantizer.dequantize(q, c)
            self._gW[n] += g

        # gradients of the biases
        if self._gb is None:
            self._gb = [0] * len(qb)

        for n, q, c in zip(itertools.count(), qb, cb):
            g = quantizer.dequantize(q, c)
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

"""
    '1-Bit Stochastic Gradient Descent and its Application to Data-Parallel Distributed Training of Speech DNNs',
    Seide, et. al. 2014
"""

import numpy as np
import itertools


def _onebit_sign_quantizer(w):
    q = np.zeros(w.shape, dtype=np.int)
    q[w > 0] = 1

    centers = np.zeros(2)

    # find the centers
    sum_q1 = np.count_nonzero(q)
    sum_q0 = q.size - np.count_nonzero(q)
    centers[1] = np.sum(w * q) / (sum_q1 + np.finfo(float).eps)
    centers[0] = np.sum(w * (1 - q)) / (sum_q0 + np.finfo(float).eps)

    return q, centers


def quantize(W):
    """
    Quantizing the given matrix with only 1 bit. The threshold is fixed to zero and the reconstruction values are computed
    to minimize the MSE. The quantization error is returned, too.
    :param W: input data (vector or ndarray) to quantize
    :return: quantized values, centers, quantization error
    """

    if W.ndim == 1:
        Q, centers = _onebit_sign_quantizer(W)

        # compute the quantization error
        E = W - centers[Q]

    else:
        Q = np.zeros(W.shape, dtype=np.int)
        centers = np.zeros((W.shape[0], 2), dtype=np.float32)
        E = np.zeros_like(W)
        # if W is an  nd array, process each column separately
        for n, w in enumerate(W):
            q, center = _onebit_sign_quantizer(w)
            Q[n, :] = q
            centers[n, :] = center
            E[n, :] = w - center[q]

    return Q, centers, E


def dequantize(Q, centers):
    """
    Dequanitze from the given 1-bit quantization and the reconstruction values.
    :param Q: input quantized values
    :param centers: centers of the quantization bins (reconstruction points)
    :return: reconstructed values
    """

    if Q.ndim == 1:
        W = centers[Q]
    else:
        W = np.zeros(Q.shape)
        for n, q, c in zip(itertools.count(), Q, centers):
            W[n] = c[q]

    return W

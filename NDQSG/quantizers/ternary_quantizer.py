"""
    Wei Wen, Cong Xu, Feng Yan, Chunpeng Wu, Yandan Wang, Yiran Chen, Hai Li,
    'TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning', NIPS 2017

    Ternary quantization:
    q = ternary_quantizer(x, scale, seed)
    Dequantizer:
    y = q * scale
"""

import numpy as np


def quantize(x, scale, seed=None):
    """
    :param x    : input vector
    :param scale: scale to normalize the input vector (infinity norm or max of all workers)
    :param seed : input seed to generate random numbers
    :return: quantized vector in {-1, 0, +1} (it is not scaled)
    """
    if seed is not None:
        np.random.seed(seed)

    # 1- normalize x
    y = np.abs(x) / scale

    # 2- create random binary numbers, b_i = 0 with probability (1-y) and b_i = 1 with probability y,
    # then multiply it with the sign(x) to get the ternary quantizer
    q = np.sign(x)
    q[np.random.random(size=q.shape) > y] = 0

    return q.astype(int)

def dequantize(q, scale):
    return q * scale

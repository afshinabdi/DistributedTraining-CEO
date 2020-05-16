"""
    Implementation of dithered quantizer and nested one dimensional quantizer
"""

import numpy as np


# =============================================================================
# dithered quantization
def quantize(W, num_levels=2, d=None):
    """
    quantize input tensor W using QSG method. the input tensor is reshaped into vector form and divided into buckets of
    length d. it used maximum value of the vector as the scaling parameter for quantization. The output scale is such that
    by multiplying it with quantized values, the points will be reconstructed.
    :param W: input tensor to be quantizer
    :param num_levels: number of levels for quantizing W, output will be in the range [-num_levels, ..., +num_levels]
    :param d: bucket size
    :return: quantized values and the scale
    """

    if d is None:
        d = W.size

    if W.size % d != 0:
        raise ValueError('the number of variables must be divisible by the bucket size (d).')

    w = np.reshape(W, newshape=(-1, d))

    # 1- normalize x
    scale = np.linalg.norm(w, ord=np.inf, axis=1) / num_levels + np.finfo(float).eps

    y = w / scale[:, np.newaxis]

    # 2- generate dither, add it to y and then quantize
    u = np.random.uniform(-0.5, 0.5, size=y.shape)
    q = np.around(y + u)  # an integer number in the range -s, ..., -1, 0, 1, ..., s

    Q = np.reshape(q, newshape=W.shape).astype(int)

    return Q, scale


def dequantize(Q, scale, d=None):
    """
    dequantize the received quantized values, usign the bucket size d and scales
    :param Q: quantized values
    :param scale: scale to multiply to the quantized values to reconstruct the original data
    :param d: bucket size
    :return: ndarray of the same shape as Q, dequantized values
    """

    if d is None:
        d = Q.size

    if Q.size % d != 0:
        raise ValueError('the number of variables must be divisible by the bucket size (d).')

    if d == Q.size:
        u = np.random.uniform(-0.5, 0.5, size=Q.shape)
        W = scale[0] * (Q - u)
    else:
        q = np.reshape(Q, (-1, d))
        u = np.random.uniform(-0.5, 0.5, size=q.shape)
        w = (q - u) * scale[:, np.newaxis]

        W = np.reshape(w, newshape=Q.shape)

    return W

# =============================================================================
# nested quantization
def nested_quantize(W, num_levels=(3, 1), d=None, alpha=1.0):
    """
    nested quantization of input tensor W. the input tensor is reshaped into vector form and divided into buckets of
    length d. it used maximum value of the vector as the scaling parameter for quantization. The output scale is such that
    by multiplying it with quantized values, the points will be reconstructed.
    :param W: input tensor to be quantizer
    :param d: bucket size
    :param num_levels: number of levels for quantizing W, output will be in the range [-num_levels, ..., +num_levels]
    :param alpha: scale for quantization
    :return: quantized values and the scale
    """

    if d is None:
        d = W.size

    if W.size % d != 0:
        raise ValueError('the number of variables must be divisible by the bucket size (d).')

    rho = num_levels[0] // num_levels[1]  # the ratio of nested quantizers
    w = np.reshape(W, (-1, d))
    # 1- normalize w
    norm_w = np.linalg.norm(w, ord=np.inf, axis=1)
    scale = norm_w / num_levels[0] + np.finfo(float).eps
    x = w / scale[:, np.newaxis]
    
    # 2- generate dither and add it to x
    u = np.random.uniform(-0.5, 0.5, size=x.shape)
    t = alpha * x + u
    q1 = np.around(t)  # Q1(t), an integer number in the range -num_levels[0], ..., -1, 0, 1, ..., num_levels[0]
    q2 = rho * np.around(t / rho)
    q = q1 - q2
    
    Q = np.reshape(q, newshape=W.shape).astype(int)

    return Q, scale


def nested_dequantize(Q, scale, d=None, alpha=1, y=0, num_levels=(3, 1)):
    """
    dequantize the received quantized values, usign the bucket size d and scales
    :param Q: quantized values
    :param scale: scale to multiply to the quantized values to reconstruct the original data
    :param d: bucket size
    :param alpha: scale parameter of the algorithm
    :param y: available side information for decoding
    :param num_levels: number of quantization levels in the range of signal
    :return: ndarray of the same shape as Q, dequantized values
    """

    if d is None:
        d = Q.size

    if Q.size % d != 0:
        raise ValueError('the number of variables must be divisible by the bucket size (d).')

    rho = num_levels[0] / num_levels[1]

    q = np.reshape(Q, (-1, d))
    y = np.reshape(y, (-1, d))

    s2 = scale * rho
    u = np.random.uniform(-0.5, 0.5, size=y.shape)
    r = (q - u) * scale[:, np.newaxis] - alpha * y
    w = y + alpha * (r - s2[:, np.newaxis] * np.around(r / s2[:, np.newaxis]))
    
    W = np.reshape(w, newshape=Q.shape)
    return W
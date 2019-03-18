import numpy as np


# =============================================================================
# compute entropy for approximate bit-rate calculations
def compute_entropy(x, alphabet_size=None):
    if alphabet_size is None:
        min_a = np.min(x)
        max_a = np.max(x)
        alphabet_size = max_a - min_a + 1
    else:
        min_a = 0

    p = np.zeros(alphabet_size)
    for n in range(alphabet_size):
        p[n] = np.count_nonzero(x == (min_a + n))

    h = -np.dot(p, np.log2((p + np.finfo(float).eps) / x.size))
    return h


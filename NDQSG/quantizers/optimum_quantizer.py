"""
   desinging optimum quantizers for different probability distributions.
"""

import itertools
import numpy as np
import scipy.stats as stat


class OptimumQuantizer:
    def __init__(self):
        self._model = None
        self._valid_levels = None
        self._quantizer_bins = None
        self._quantizer_centers = None

    def initialize_quantizer(self, model, num_levels=(2, 4, 8), sparsity_thr=1e-4, x=None):
        self._model = model
        self._valid_levels = np.array(num_levels)
        self._quantizer_bins = [None] * len(self._valid_levels)
        self._quantizer_centers = [None] * len(self._valid_levels)

        if model == 'normal' or model == 'n':
            self._initialize_normal_quantizer()
        elif model == 'sparse-normal' or model == 'sn':
            self._initialize_sparse_normal_quantizer(sparsity_thr)
        elif model == 'folded-normal' or model == 'fn':
            self._initialize_folded_normal_quantizer()
        elif model == 'sparse-folded-normal' or model == 'sfn':
            self._initialize_sparse_folded_normal_quantizer(sparsity_thr)
        elif model == 'uniform' or model == 'u':
            self._initialize_uniform_quantizer()
        elif model == 'sparse-uniform' or model == 'su':
            self._initialize_sparse_uniform_quantizer(sparsity_thr)
        elif model == 'empirical' or model == 'e':
            self._initialize_empirical_quantizer(x)
        else:
            raise ValueError('Unknown data distribution model!')


    def quantize(self, x, num_levels):
        if num_levels not in self._valid_levels:
            raise ValueError('Quantizer for the given number of levels has not been initialized.')

        q_idx = np.where(self._valid_levels == num_levels)[0][0]

        q = np.digitize(x, self._quantizer_bins[q_idx])
        
        return q, self._quantizer_centers[q_idx]


    def dequantize(self, q, num_levels):
        if num_levels not in self._valid_levels:
            raise ValueError('Quantizer for the given number of levels has not been initialized.')

        q_idx = np.where(self._valid_levels == num_levels)[0][0]
        x = self._quantizer_centers[q_idx][q]

        return x

    # =========================================================================
    # using Lloyd-Max algorithm, find the optimum quantizer for different distributions
    def _initialize_normal_quantizer(self):
        s = np.sqrt(2*np.pi)

        max_iterations = 1000
        for n, num_levels in enumerate(self._valid_levels):
            # initialize quantizer's thresholds and centers
            bins = np.linspace(-1, 1, num_levels + 1)
            centers = (bins[1:] + bins[:-1]) / 2
            bins = bins[1:-1]

            for _ in range(max_iterations):
                old_centers = centers.copy()
                cdf_x = stat.norm.cdf(bins)
                exp_x = -np.exp(-bins**2 / 2) / s

                # a- updating centers
                centers[0] =  exp_x[0] / cdf_x[0]
                centers[1:-1] = (exp_x[1:] - exp_x[0:-1]) / (cdf_x[1:] - cdf_x[0:-1])
                centers[-1] = -exp_x[-1] / (1-cdf_x[-1])

                # b- update bins
                bins = (centers[:-1] + centers[1:]) / 2

                # c- check for convergence
                if np.max(np.abs(centers - old_centers)) < 1e-3:
                    break

            self._quantizer_bins[n] = bins
            self._quantizer_centers[n] = centers

    def _initialize_sparse_normal_quantizer(self, thr):
        s = np.sqrt(2*np.pi)

        max_iterations = 1000
        for n, num_levels in enumerate(self._valid_levels):
            # initialize quantizer's thresholds and centers
            K = 1 + num_levels // 2
            bins = np.linspace(thr, 1, K)
            bins = np.concatenate((np.linspace(-1, -thr, K), np.linspace(thr, 1, K)))
            centers = (bins[1:] + bins[:-1]) / 2
            bins = bins[1:-1]

            for _ in range(max_iterations):
                old_centers = centers.copy()
                cdf_x = stat.norm.cdf(bins)
                exp_x = -np.exp(-bins**2 / 2) / s

                # a- updating centers
                centers[0] =  exp_x[0] / cdf_x[0]
                centers[1:-1] = (exp_x[1:] - exp_x[0:-1]) / (cdf_x[1:] - cdf_x[0:-1])
                centers[-1] = -exp_x[-1] / (1-cdf_x[-1])

                # b- update bins
                bins = (centers[:-1] + centers[1:]) / 2
                bins[K - 2] = -thr
                bins[K - 1] = thr

                # c- check for convergence
                if np.max(np.abs(centers - old_centers)) < 1e-3:
                    break

            self._quantizer_bins[n] = bins
            self._quantizer_centers[n] = centers

    def _initialize_folded_normal_quantizer(self):
        s = np.sqrt(2 / np.pi)

        max_iterations = 1000
        for n, num_levels in enumerate(self._valid_levels):
            # initialize quantizer's thresholds and centers
            bins = np.linspace(0, 1, num_levels + 1)
            centers = (bins[1:] + bins[:-1]) / 2
            bins = bins[1:-1]

            for _ in range(max_iterations):
                old_centers = centers.copy()
                cdf_x = 2 * stat.norm.cdf(bins) - 1
                mean_x = s * (1 - np.exp(-bins**2 / 2))

                # a- updating centers
                centers[0] =  mean_x[0] / cdf_x[0]
                centers[1:-1] = (mean_x[1:] - mean_x[0:-1]) / (cdf_x[1:] - cdf_x[0:-1])
                centers[-1] = (s - mean_x[-1]) / (1-cdf_x[-1])

                # b- update bins
                bins = (centers[:-1] + centers[1:]) / 2

                # c- check for convergence
                if np.max(np.abs(centers - old_centers)) < 1e-3:
                    break

            self._quantizer_bins[n] = bins
            self._quantizer_centers[n] = centers

    def _initialize_sparse_folded_normal_quantizer(self, thr):
        s = np.sqrt(2 / np.pi)

        max_iterations = 1000
        for n, num_levels in enumerate(self._valid_levels):
            # initialize quantizer's thresholds and centers
            bins = np.linspace(thr, 1, num_levels + 1)
            centers = np.concatenate(([0], (bins[1:] + bins[:-1]) / 2))
            bins = bins[:-1]

            for _ in range(max_iterations):
                old_centers = centers.copy()
                cdf_x = 2 * stat.norm.cdf(bins) - 1
                mean_x = s * (1 - np.exp(-bins**2 / 2))

                # a- updating centers
                centers[1:-1] = (mean_x[1:] - mean_x[0:-1]) / (cdf_x[1:] - cdf_x[0:-1])
                centers[-1] = (s - mean_x[-1]) / (1-cdf_x[-1])

                # b- update bins
                bins = (centers[:-1] + centers[1:]) / 2
                bins[0] = thr

                # c- check for convergence
                if np.max(np.abs(centers - old_centers)) < 1e-3:
                    break

            self._quantizer_bins[n] = bins
            self._quantizer_centers[n] = centers

    def _initialize_uniform_quantizer(self):
         for n, num_levels in enumerate(self._valid_levels):
            bins = np.linspace(0, 1, num_levels + 1)
            centers = (bins[1:] + bins[:-1]) / 2
            bins = bins[1:-1]

            self._quantizer_bins[n] = bins
            self._quantizer_centers[n] = centers
       
    def _initialize_sparse_uniform_quantizer(self, thr):
        for n, num_levels in enumerate(self._valid_levels):
            bins = np.linspace(thr, 1, num_levels + 1)
            bins = np.concatenate(([-thr], bins))
            centers = (bins[1:] + bins[:-1]) / 2
            bins = bins[1:-1]

            self._quantizer_bins[n] = bins
            self._quantizer_centers[n] = centers

    def _initialize_empirical_quantizer(self, X):
        x = np.reshape(X, newshape=-1)
        min_x = np.min(x)
        max_x = np.max(x)

        for n, num_levels in enumerate(self._valid_levels):
            # initialize bins
            bins = np.linspace(min_x, max_x, num_levels + 1)
            centers = (bins[:-1] + bins[1:]) / 2
            bins = bins[1:-1]

            for _ in range(1000):
                centers_old = centers.copy()
                # quantize input vector
                q = np.digitize(x, bins)
                _optimize_centers_average(x, q, centers, num_levels)
                bins = (centers[1:] + centers[:-1]) / 2

                if np.max(np.abs(centers - centers_old)) < 1e-3:
                    break

            self._quantizer_bins[n] = bins
            self._quantizer_centers[n] = centers


# =============================================================================
# optimize quantizer's reconstruction points by averaging the points in each bin
def _optimize_centers_average(w, q, center, num_levels):
    for n in range(num_levels):
        if n in q:
            center[n] = np.mean(w[q == n])

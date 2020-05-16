"""
    Simulation of 'dithered quantized stochastic gradient' with distributed source coding
"""

import numpy as np
import source_coding.arithmetic_codec as ac_codec

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
        self._alphabet_size = 4

        # gradients and the quantized values
        self._gW = None
        self._gb = None
        self._sW = None
        self._sb = None

    # _________________________________________________________________________
    # set parameters of the quantizer
    def set_quantizer(self, seed, clip_thr=None, bucket_size=None, num_levels=1):
        self._seed = seed
        self._clip_thr = clip_thr
        self._bucket_size = bucket_size
        self._num_levels = num_levels
        # make sure alphabet size is a power of 2
        self._alphabet_size = 2 * self._num_levels + 1
        self._alphabet_size = int(2 ** (np.ceil(np.log2(self._alphabet_size))))

    def set_codec(self, dsc_engine):
        self._codec = dsc_engine

    # _________________________________________________________________________
    # compute the gradients
    def compute_gradients(self, x, y):
        self._gW, self._gb = self._nn.compute_gradients(x, y)

        if self._clip_thr is not None:
            for n in range(len(self._gW)):
                thr = self._clip_thr * np.std(self._gW[n])
                self._gW[n] = np.clip(self._gW[n], -thr, thr)

            for n in range(len(self._gb)):
                thr = self._clip_thr * np.std(self._gb[n])
                self._gb[n] = np.clip(self._gb[n], -thr, thr)

    # _________________________________________________________________________
    # find the scaling factor for each bucket, sharing it with other workers and set the value
    def get_scale_factors(self):
        if self._bucket_size is None:
            self._bucket_size = [[v.size for v in self._gW], [v.size for v in self._gb]]

        # compute the scale factor for the gradients of the weights
        d = self._bucket_size[0]
        s_gW = [0] * len(self._gW)
        for n, g in enumerate(self._gW):
            g = np.reshape(g, newshape=(-1, d[n]))
            s_gW[n] = (np.linalg.norm(g, ord=np.inf, axis=1) + np.finfo('float').eps) / self._num_levels

        # compute the scale factor for the gradients of the biases
        d = self._bucket_size[1]
        s_gb = [0] * len(self._gb)
        for n, g in enumerate(self._gb):
            g = np.reshape(g, newshape=(-1, d[n]))
            s_gb[n] = (np.linalg.norm(g, ord=np.inf, axis=1) + np.finfo('float').eps) / self._num_levels

        return s_gW, s_gb

    def set_scale_factors(self, sW, sb):
        self._sW = sW
        self._sb = sb

    # _________________________________________________________________________
    # compress the gradients
    def get_compressed_gradients(self, use_dsc=False):
        code_length = 0
        q_gW, q_gb = self._quantize_gradients()

        # encode the quantized weights
        w_codes = []
        for n, v in enumerate(q_gW):
            q = np.reshape(v + self._num_levels, -1).astype(np.uint8)

            if use_dsc:
                code, code_len = self._codec[n].encode(q)
            else:
                code = ac_codec.adaptive_encoder(input=q, alphabet_size=self._alphabet_size)
                code_len = len(code) * 8

            w_codes = w_codes + [code]
            code_length += code_len

        # encode the biases using adaptive arithmetic coding
        b_codes = []
        for n, v in enumerate(q_gb):
            q = np.reshape(v + self._num_levels, -1).astype(np.uint8)
            code = ac_codec.adaptive_encoder(input=q, alphabet_size=self._alphabet_size)
            code_len = len(code) * 8

            b_codes = b_codes + [code]
            code_length += code_len

        return w_codes, b_codes, code_length

    # =========================================================================
    # apply the received gradient (from server, ...) to the neural network
    def apply_gradients(self, gW, gb):
        self._nn.apply_gradients(gW, gb)

    # _________________________________________________________________________
    # dither quatization of the gradients
    def _quantize_gradients(self):
        np.random.seed(self._seed)

        q_gW = [0] * len(self._gW)
        for n, g in enumerate(self._gW):
            q_gW[n] = self._dither_qauntizer(g, scale=self._sW[n], bucket_size=self._bucket_size[0][n])

        q_gb = [0] * len(self._gb)
        for n, g in enumerate(self._gb):
            q_gb[n] = self._dither_qauntizer(g, scale=self._sb[n], bucket_size=self._bucket_size[1][n])

        self._seed = np.random.randint(min_seed, max_seed)

        return q_gW, q_gb

    def _dither_qauntizer(self, X, scale, bucket_size):
        x = np.reshape(X, newshape=(-1, bucket_size))

        # 1- normalize x
        y = x / scale[:, np.newaxis]

        # 2- generate dither, add it to y and then quantize
        u = np.random.uniform(-0.5, 0.5, size=y.shape)
        q = np.around(y + u).astype(int)  # an integer number in the range -s, ..., -1, 0, 1, ..., s

        return q


class AggregationNode:
    def __init__(self, num_workers, layer_shapes):

        self._num_received = 0
        self._num_workers = num_workers
        self._num_layers = len(layer_shapes) - 1
        self._layer_shapes = [[layer_shapes[n], layer_shapes[n+1]] for n in range(self._num_layers)]

        # parameters of each worker's quantizer
        self._seeds = [default_seed] * num_workers
        self._bucket_size = None
        self._num_levels = 1
        self._max_si_levels = 3
        self._alphabet_size = 4

        # compression engine of the workers
        self._dsc_codec = [None] * num_workers
        self._si = [None] * self._num_layers

        # quantized gradients of the workers, used to compute side information
        self._qW = [[0] * self._num_layers for _ in range(num_workers)]

        self._gW = [np.zeros(np.prod(s)) for s in self._layer_shapes]
        self._gb = [np.zeros(s[-1]) for s in self._layer_shapes]

        # scale factor
        self._sW = None
        self._sb = None

    # =========================================================================
    # set parameters of the quantizer for all workers
    def set_quantizer(self, seeds, bucket_size=None, num_levels=1, max_si_levels=3):
        self._seeds = seeds
        self._bucket_size = bucket_size
        self._num_levels = num_levels
        self._max_si_levels = max_si_levels
        # make sure alphabet size is a power of 2
        self._alphabet_size = 2 * self._num_levels + 1
        self._alphabet_size = int(2 ** (np.ceil(np.log2(self._alphabet_size))))

        if self._bucket_size is None:
            self._bucket_size = [[int(np.prod(v)) for v in self._layer_shapes], [int(v[-1])
                                                                                 for v in self._layer_shapes]]

    def set_codec(self, dsc_codec):
        self._dsc_codec = dsc_codec

    # =========================================================================
    # reset the parameters of the algorithm
    def reset_node(self):
        self._num_received = 0
        # the gradients are stored in vectorized format
        self._gW = [np.zeros(np.prod(s)) for s in self._layer_shapes]
        self._gb = [np.zeros(s[-1]) for s in self._layer_shapes]

    def set_scale_factors(self, sW, sb):
        self._sW = sW
        self._sb = sb

    # update the side information after receiving data from group 1 and before group 2
    def update_side_information(self, si_workers):
        assert self._num_received > 0, 'no data has been received from the workers to compute the side information!'

        num_si_levels = self._num_levels * len(si_workers)
        if num_si_levels > self._max_si_levels:
            si_scale = self._max_si_levels / num_si_levels
            num_si_levels = self._max_si_levels
        else:
            si_scale = 1

        a_y = 2 * num_si_levels + 1
        bins_x = np.arange(-0.5, self._alphabet_size, 1)
        bins_y = np.arange(-0.5, a_y, 1)

        pxy = [0] * self._num_layers
        py = [0] * self._num_layers
        for w_id in si_workers:
            qW = self._qW[w_id]

            for n in range(self._num_layers):
                self._si[n] = self._gW[n]*si_scale
                p, _, _ = np.histogram2d(qW[n], self._si[n], bins=[bins_x, bins_y])
                pxy[n] += p

        for n in range(self._num_layers):
            pxy[n] += 1
            py[n] = np.sum(pxy[n], axis=0)

            pxy[n] /= py[n][np.newaxis, :]
            py[n] /= np.sum(py)

        # update the DISCUS codecs
        for n in range(self._num_layers):
            self._dsc_codec[n].set_distributions(pxy[n], py[n])

    # _________________________________________________________________________
    # receive compressed gradients from group 1 which uses AAC
    def receive_compressed_gradients_grp1(self, worker_id, w_codes, b_codes):
        self._num_received += 1
        # decode the weights using adaptive arithmetic codec
        for n, code in enumerate(w_codes):
            self._qW[worker_id][n] = ac_codec.adaptive_decoder(
                input=code, decode_len=int(np.prod(self._layer_shapes[n])), alphabet_size=self._alphabet_size)

            self._gW[n] += self._qW[worker_id][n]

        # decode the biases using adaptive arithmetic codec
        for n, code in enumerate(b_codes):
            dec = ac_codec.adaptive_decoder(
                input=code, decode_len=int(self._layer_shapes[n][-1]), alphabet_size=self._alphabet_size)
            self._gb[n] += dec

    # receive compressed gradients from group 2 which may use DISCUS
    def receive_compressed_gradients_grp2(self, worker_id, w_codes, b_codes, use_dsc=False):
        self._num_received += 1
        if use_dsc:
            # decode the weights using DISCUS
            for n, code in enumerate(w_codes):
                self._qW[worker_id][n] = self._dsc_codec[n].decode(code, self._si[n])
                self._gW[n] += self._qW[worker_id][n]
        else:
            # decode the weights using adaptive arithmetic codec
            for n, code in enumerate(w_codes):
                self._qW[worker_id][n] = ac_codec.adaptive_decoder(
                    input=code, decode_len=int(np.prod(self._layer_shapes[n])), alphabet_size=self._alphabet_size)
                self._gW[n] += self._qW[worker_id][n]

        # decode the biases using adaptive arithmetic codec
        for n, code in enumerate(b_codes):
            dec = ac_codec.adaptive_decoder(
                input=code, decode_len=int(self._layer_shapes[n][-1]), alphabet_size=self._alphabet_size)
            self._gb[n] += dec  # no need to reshape, the biase is a vector

    # _________________________________________________________________________
    # get computed and quantized SG from a worker and aggregate the gradient
    def get_aggregated_gradients(self):
        avg_uW, avg_ub = self._compute_average_dither()

        if self._num_received == 0:
            return avg_uW, avg_ub

        # dequantization   normalize the sum of the received quantized gradients
        for n in range(self._num_layers):
            u = avg_uW[n].reshape((-1, self._bucket_size[0][n]))
            g = self._gW[n].reshape((-1, self._bucket_size[0][n]))
            g = (g / self._num_workers - self._num_levels - u) * self._sW[n][:, np.newaxis]
            self._gW[n] = np.reshape(g, newshape=self._layer_shapes[n])

            u = avg_ub[n].reshape((-1, self._bucket_size[1][n]))
            g = self._gb[n].reshape((-1, self._bucket_size[1][n]))
            g = (g / self._num_workers - self._num_levels - u) * self._sb[n][:, np.newaxis]
            self._gb[n] = np.reshape(g, newshape=self._layer_shapes[n][-1])

        return self._gW, self._gb

    # _________________________________________________________________________
    # compute the average of the dither signals of all workers and update the seeds
    def _compute_average_dither(self):
        avg_uW = [0] * len(self._gW)
        avg_ub = [0] * len(self._gb)

        for w_id in range(self._num_workers):
            # generate random uniform dither
            np.random.seed(self._seeds[w_id])
            for n, g in enumerate(self._gW):
                avg_uW[n] += np.random.uniform(-0.5, 0.5, size=g.size)

            for n, g in enumerate(self._gb):
                avg_ub[n] += np.random.uniform(-0.5, 0.5, size=g.size)

            # update the seed numebrs
            self._seeds[w_id] = np.random.randint(min_seed, max_seed)

        # reshape and scale the dither signals
        avg_uW = [np.reshape(v, (-1, d)) / self._num_workers for v, d in zip(avg_uW, self._bucket_size[0])]
        avg_ub = [np.reshape(v, (-1, d)) / self._num_workers for v, d in zip(avg_ub, self._bucket_size[1])]

        return avg_uW, avg_ub

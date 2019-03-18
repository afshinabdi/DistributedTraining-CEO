import numpy as np
import source_coding.discus.ldpc_design as code_design
import source_coding.discus.binaryDSC as dsc_binary_codec
import source_coding.discus.nonbinaryDSC as dsc_nonbinary_codec

class DISCUSEngine:
    def __init__(self):
        self._num_chks = -1
        self._pxy = None
        self._py = None

        self._dsc = None
        self._ldpc_codes = None
        self._alphabet_size = 2
        self._bits_per_symbol = 1

    def initialize(self, alphabet_size, ldpc_codes):
        self._alphabet_size = alphabet_size
        self._ldpc_codes = ldpc_codes

        # check if alphabet size is a power of 2
        if (alphabet_size < 2) or ((alphabet_size & (alphabet_size - 1)) != 0):
            alphabet_size = int(2 ** (np.ceil(np.log2(alphabet_size))))
            self._alphabet_size = alphabet_size

        # initialize the probability distributions
        self._bits_per_symbol = int(np.log2(self._alphabet_size).item())


        if self._alphabet_size == 2:
            self._dsc = dsc_binary_codec.BinaryDSC()
        else:
            self._dsc = dsc_nonbinary_codec.NonbinaryDSC()

        self.set_distributions()


    # =========================================================================
    # set the parameters of the DISCUS algorithm
    def set_distributions(self, pxy=None, py=None):
        if pxy is None:
            pxy = np.ones(shape=[self._alphabet_size, self._alphabet_size], dtype=np.float) / self._alphabet_size

        if py is None:
            py = np.ones(shape=self._alphabet_size, dtype=np.float) / self._alphabet_size

        self._pxy = pxy
        self._py = py

        h = -np.dot(np.sum(self._pxy * np.log2(self._pxy + 1e-12), axis=0), self._py) / np.log2(self._alphabet_size)

        num_vars = self._ldpc_codes['N']
        if h > 0.9:
            num_checks = num_vars
            var_index = np.arange(0, num_vars).astype(np.uint32)
            chk_index = np.arange(0, num_checks).astype(np.uint32)
        else:
            num_checks = int(1.1 * h * num_vars)
            index = np.where(self._ldpc_codes['M'] > num_checks)
            idx = np.argmin(self._ldpc_codes['M'][index])
            index = index[0][idx]
            num_checks = self._ldpc_codes['M'][index].item(0)
            chk_name = 'chk_index_{}'.format(index)
            var_name = 'var_index_{}'.format(index)
            chk_index = self._ldpc_codes[chk_name]
            var_index = self._ldpc_codes[var_name]

        if (num_checks != self._num_chks):
            self._num_chks = num_checks
            # new code should be used
            if self._alphabet_size == 2:
                self._dsc.initialize(num_checks, num_vars, chk_index, var_index)
            else:
                h_values = np.random.randint(1, self._alphabet_size, size=len(chk_index)).astype(np.uint8)
                self._dsc.initialize(self._bits_per_symbol, num_checks, num_vars, chk_index, var_index, h_values)

    # =========================================================================
    # get the alphabet size of the codec
    @property
    def alphabet_size(self):
        return self._alphabet_size

    # =========================================================================
    # encoding function
    def encode(self, x):
        code = self._dsc.encode(input=x)
        code_len = self._bits_per_symbol * code.size

        return code, code_len

    # =================================================================================================================
    # decoding function
    def decode(self, code, si_sequence):
        x = self._dsc.decode(input_code=code, si_sequence=si_sequence.astype(np.uint8), px_y=self._pxy, max_iter=100)

        return x

"""
   evaluating discus method for distributed training
"""
import os
import scipy.io as sio
import scipy.stats as st
import numpy as np
import tensorflow as tf

from FullyConnected import FCModel
from mnist_dataset import MNISTDataset
import compression as cmp
import discus_engine as dsc_engine
import ldpc_design as ldpc_design
import CodeMaker as code_maker
import distributed_dsc as dt_dsc

training_epochs = 5
total_batch_size = 200
training_algorithm = 'GD'
learning_rate = 0.01
decay_rate = 0.9
epochs_per_decay = 1
iter_per_epoch = 60000 // total_batch_size
decay_step = epochs_per_decay * iter_per_epoch
max_iterations = training_epochs * iter_per_epoch

output_folder = 'D:/Simulations/DistributedDataTraining/CEO/FC/'
output_folder = output_folder + training_algorithm + '/exp{}/'
mnist_folder = 'D:/DataBase/MNIST/raw'
ldpc_filename = 'D:/Simulations/DistributedDataTraining/CEO/FC/ldpc_codes_N{}.mat'


def generate_random_parameters(layer_shapes):
    num_layers = len(layer_shapes) - 1

    w0 = [0] * num_layers
    b0 = [0] * num_layers
    # create initial parameters for the network
    for n in range(num_layers):
        w0[n] = st.truncnorm(-2, 2, loc=0, scale=0.1).rvs((layer_shapes[n], layer_shapes[n + 1]))
        b0[n] = np.ones(layer_shapes[n + 1]) * 0.1

    return w0, b0


def evaluate_basemodel(nn_params, seed):
    mnist_db = MNISTDataset()
    mnist_db.create_dataset(mnist_folder, True, True, validation_size=0, seed=seed)
    test_images, test_labels = mnist_db.test

    nn = FCModel()
    nn.create_network(nn_params)
    nn.initialize()

    accuracy = np.zeros(max_iterations)
    for n in range(max_iterations):
        x, y = mnist_db.next_batch(total_batch_size)
        nn.train(x, y)
        accuracy[n] = nn.accuracy(test_images, test_labels)
        if n % 50 == 0:
            print('{0:03d}: learning rate={1:.4f}, accuracy={2:.2f}'.format(n, nn.learning_rate(), accuracy[n] * 100))

    # final weights of the trained nn
    wf, bf = nn.get_weights()
    parameter_size = np.sum([v.size for v in wf]) + np.sum([v.size for v in bf])
    raw_rate = parameter_size * 32

    return accuracy, raw_rate, wf, bf


# =============================================================================
# designs different LDPC matrices with different rates for the given block_length
def design_gallager_ldpc_codes(block_length, rates):
    num_checks = []
    ldpc_codes = {'N': block_length}
    valid_dc = []
    for dc in range(4, 20):
        if block_length % dc == 0:
            valid_dc += [dc]
    
    valid_dc = np.array(valid_dc)

    for (n, r) in enumerate(rates):
        if r > 0.9:
            code_length = block_length
            var_index = np.arange(0, block_length).astype(np.uint32)
            chk_index = np.arange(0, code_length).astype(np.uint32)
        else:
            code_length = int(block_length * r)
            dv = code_length * valid_dc / block_length
            idx = np.argmin(np.abs(dv-np.around(dv)))
            dv = int(dv[idx])
            dc = valid_dc[idx]

            H = code_maker.make_H_gallager(block_length, dv, dc)
            code_length = len(H)
            I = []
            J = []
            for (r, h) in enumerate(H):
                I.extend([r] * len(h))
                J.extend(h)
                
            chk_index = np.array(I, dtype=np.uint32)
            var_index = np.array(J, dtype=np.uint32)

        num_checks.append(code_length)
        chk_name = 'chk_index_{}'.format(n)
        var_name = 'var_index_{}'.format(n)

        ldpc_codes[chk_name] = chk_index
        ldpc_codes[var_name] = var_index

    ldpc_codes['M'] = np.array(num_checks)
    return ldpc_codes


def design_ldpc_codes(block_length, rates):
    num_checks = []
    ldpc_codes = {'N': block_length}
    for (n, r) in enumerate(rates):
        if r > 0.9:
            code_length = block_length
            var_index = np.arange(0, block_length).astype(np.uint32)
            chk_index = np.arange(0, code_length).astype(np.uint32)
        else:
            code_length = int(block_length * r)
            chk_index, var_index = ldpc_design.peg_design(code_length, block_length, [2, 3, 7],
                                                          [0.521814, 0.271293, 0.206893])

        num_checks.append(code_length)
        chk_name = 'chk_index_{}'.format(n)
        var_name = 'var_index_{}'.format(n)

        ldpc_codes[chk_name] = chk_index
        ldpc_codes[var_name] = var_index

    ldpc_codes['M'] = np.array(num_checks)
    return ldpc_codes


def generate_ldpc_codes(msg_lengths):
    ldpc_codes = [0] * len(msg_lengths)
    for n, msg_len in enumerate(msg_lengths):
        fname = ldpc_filename.format(msg_len)

        if os.path.isfile(fname):
            codes = sio.loadmat(fname)
        else:
            # codes = design_ldpc_codes(msg_len, [0.9, 0.8, 0.7, 0.6, 0.5])
            codes = design_gallager_ldpc_codes(msg_len, [0.9, 0.8, 0.7, 0.6, 0.5])
            sio.savemat(fname, mdict=codes)

        # =========================================================================
        # remove the redundant dimension in codes
        codes['N'] = codes['N'].item()
        codes['M'] = codes['M'].squeeze()
        for m in range(codes['M'].size):
            chk_name = 'chk_index_{}'.format(m)
            var_name = 'var_index_{}'.format(m)

            codes[chk_name] = codes[chk_name].squeeze().astype(np.uint32)
            codes[var_name] = codes[var_name].squeeze().astype(np.uint32)

        ldpc_codes[n] = codes
    
    return ldpc_codes


def evaluate_dsc(num_workers, nn_params, dqsg_params, dsc_ratio, seed):
    # compute number of weights, each layer is compressed separately
    layer_shapes = np.array(nn_params.get('layer_shapes', [784, 300, 100, 10]))
    msg_lengths = list(layer_shapes[1:] * layer_shapes[:-1])
    num_layers = len(layer_shapes) - 1
    
    # design ldpc codes
    ldpc_codes = generate_ldpc_codes(msg_lengths)

    # create database
    mnist_db = MNISTDataset()
    mnist_db.create_dataset(mnist_folder, True, True, validation_size=0, seed=seed)
    test_images, test_labels = mnist_db.test

    # create neural network model
    nn_model = FCModel()
    nn_model.create_network(nn_params)
    nn_model.initialize()

    # create workers and server
    workers = [dt_dsc.WorkerNode(nn_model) for _ in range(num_workers)]
    server = dt_dsc.AggregationNode(num_workers, layer_shapes)

    # 1- quantization parameters
    clip_thr = dqsg_params.get('gradient-clip', None)
    num_levels = dqsg_params.get('num-levels', 1)
    bucket_size = dqsg_params.get('bucket-size', None)
    dt_seeds = np.random.randint(dt_dsc.min_seed, dt_dsc.max_seed, size=num_workers)

    server.set_quantizer(dt_seeds, bucket_size, num_levels, max_si_levels=4)
    for w_id in range(num_workers):
        workers[w_id].set_quantizer(dt_seeds[w_id], clip_thr, bucket_size, num_levels)
        
    # 2- compression parameters
    alphabet_size = 2 * num_levels + 1
    dsc_engines = None

    n = int(num_workers * dsc_ratio + 0.5)
    group1 = np.arange(0, n)  # uses AAC
    group2 = np.arange(n, num_workers)  # uses DISCUS
    dsc_codecs = [dsc_engine.DISCUSEngine() for _ in range(num_layers)]
    for k in range(num_layers):
        dsc_codecs[k].initialize(alphabet_size, ldpc_codes[k])

    server.set_codec(dsc_codecs)
    for w_id in group2:
        workers[w_id].set_codec(dsc_codecs)


    accuracy = np.zeros(max_iterations)
    code_rate = np.zeros(max_iterations)

    batch_size = total_batch_size // num_workers
    for n in range(max_iterations):
        use_dsc = n > 10
        x, y = mnist_db.next_batch(total_batch_size)

        rec_rate = 0
        server.reset_node()
        # 1- compute gradients
        for k in range(num_workers):
            x_batch = x[k * batch_size:(k + 1) * batch_size]
            y_batch = y[k * batch_size:(k + 1) * batch_size]
            workers[k].compute_gradients(x_batch, y_batch)

        # 2- computing maximum of scale factors and use it for all workers
        sW, sb = workers[0].get_scale_factors()
        for k in range(1, num_workers):
            _sW, _sb = workers[k].get_scale_factors()
            sW = [np.maximum(v1, v2) for v1, v2 in zip(sW, _sW)]
            sb = [np.maximum(v1, v2) for v1, v2 in zip(sb, _sb)]

        server.set_scale_factors(sW, sb)
        for k in range(num_workers):
            workers[k].set_scale_factors(sW, sb)

        # 3- receive compressed gradients from group 1
        for k in group1:
            w_codes, b_codes, code_lengths = workers[k].get_compressed_gradients(use_dsc=False)
            server.receive_compressed_gradients_grp1(k, w_codes, b_codes)

            code_rate[n] += code_lengths

        if use_dsc:
            # update side information
            server.update_side_information(si_workers=group1)

        # 4- receive compressed gradients from group 2
        for k in group2:
            w_codes, b_codes, code_lengths = workers[k].get_compressed_gradients(use_dsc=use_dsc)
            server.receive_compressed_gradients_grp2(k, w_codes, b_codes, use_dsc)

            code_rate[n] += code_lengths

        # 5- aggregate the gradients
        gW, gb = server.get_aggregated_gradients()
        nn_model.apply_gradients(gW, gb)

        accuracy[n] = nn_model.accuracy(test_images, test_labels)
        if n % 50 == 0:
            print('{0:03d}: learning rate={1:.4f}, accuracy={2:.2f}'.format(n, nn_model.learning_rate(), accuracy[n] * 100))

    wf, bf = nn_model.get_weights()

    return accuracy, code_rate, wf, bf


def main():
    layer_shapes = (784, 300, 100, 10)
    bucket_size = [[300, 300, 200], [300, 100, 10]]

    nn_params = {'layer_shapes': layer_shapes,
                 'training_alg': training_algorithm,
                 'learning_rate': learning_rate,
                 'decay_rate': decay_rate,
                 'decay_step': decay_step,
                 'compute_gradients': True,
                 }

    dqsg_params = {'gradient-clip': None, 'num-levels': 3, 'bucket-size': bucket_size}
    dsc_ratio = 0.5

    for expr in [0]:  # range(3):

        print('=' * 80)
        print('Running experiment {0}'.format(expr))

        print('Training base model to compare ...')
        result_folder = output_folder.format(expr)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        file_name = result_folder + 'base_model.mat'
        if not os.path.exists(file_name):
            seed = np.random.randint(1000, 1000000)
            w0, b0 = generate_random_parameters(layer_shapes)
            nn_params['initial_w'] = w0
            nn_params['initial_b'] = b0

            base_accuracy, base_rate, w_base, b_base = evaluate_basemodel(nn_params, seed)
            file_name = result_folder + 'initial_weights.npz'
            np.savez(file_name, *w0, *b0, seed)

            file_name = result_folder + 'base_model.mat'
            sio.savemat(file_name, mdict={'acc': base_accuracy, 'rate': base_rate, 'w': w_base, 'b': b_base})
        else:
            # load initial weights
            file_name = result_folder + 'initial_weights.npz'
            data = np.load(file_name)
            num_layers = len(data.keys()) // 2
            w_keys = ['arr_{}'.format(n) for n in range(num_layers)]
            b_keys = ['arr_{}'.format(n) for n in range(num_layers, 2 * num_layers)]
            s_key = 'arr_{}'.format(2 * num_layers)
            nn_params['initial_w'] = [data[k] for k in w_keys]
            nn_params['initial_b'] = [data[k] for k in b_keys]
            seed = data[s_key]

        for number_workers in [4, 8, 16]:
            print('=' * 40)
            print('Running for {0} workers'.format(number_workers))
            result_folder = output_folder.format(expr) + '{}/'.format(number_workers)
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
          
            print('testing dithered quantized SG with DSC method with 1 level, no bucket...')
            dqsg_params = {'gradient-clip': None, 'num-levels': 1, 'bucket-size': None}
            accuracy, code_rate, w, b = evaluate_dsc(number_workers, nn_params, dqsg_params, dsc_ratio, seed)
            file_name = result_folder + 'dsc_nb_1.mat'
            sio.savemat(file_name, mdict={'acc': accuracy, 'H': code_rate, 'w': w, 'b': b})

            print('testing nested dithered quantized SG with DSC method with (3, 1) level, with bucket...')
            dqsg_params = {'gradient-clip': None, 'num-levels': 1, 'bucket-size': bucket_size}
            accuracy, code_rate, w, b = evaluate_dsc(number_workers, nn_params, dqsg_params, dsc_ratio, seed)
            file_name = result_folder + 'dsc_wb_1.mat'
            sio.savemat(file_name, mdict={'acc': accuracy, 'H': code_rate, 'w': w, 'b': b})


if __name__ == '__main__':
    main()

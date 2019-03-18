import os
import scipy.io as sio
import scipy.stats as st
import numpy as np
import tensorflow as tf

from Lenet import LenetModel
from mnist_dataset import MNISTDataset
import compression as cmp
import distributed_1bit as dt_1bit
import distributed_qsg as dt_qsg
import distributed_ndqsg as dt_ndqsg

training_epochs = 5
total_batch_size = 200
training_algorithm = 'GD'
learning_rate = 0.01
decay_rate = 0.9
epochs_per_decay = 1
iter_per_epoch = 60000 // total_batch_size
decay_step = epochs_per_decay * iter_per_epoch
max_iterations = training_epochs * iter_per_epoch

output_folder = '../../data/DistributedDataTraining/CEO/Lenet/NDQSG/'
output_folder = output_folder + training_algorithm + '/exp{}/'
mnist_folder = '../../data/DataBase/MNIST/raw'


def generate_random_parameters(layer_shapes):
    layer_shapes = [[5, 5, 1, 32], [5, 5, 32, 64], [7 * 7 * 64, 512], [512, 10]]
    num_layers = 4

    w0 = [0] * num_layers
    b0 = [0] * num_layers
    # create initial parameters for the network
    for n in range(num_layers):
        w0[n] = st.truncnorm(-2, 2, loc=0, scale=0.1).rvs(layer_shapes[n])
        b0[n] = np.ones(layer_shapes[n][-1]) * 0.1

    return w0, b0


def evaluate_basemodel(nn_params, seed):
    mnist_db = MNISTDataset()
    mnist_db.create_dataset(mnist_folder, False, False, validation_size=0, seed=seed)
    test_images, test_labels = mnist_db.test

    nn = LenetModel()
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


def evaluate_1bit(num_workers, nn_params, seed):
    # create database
    mnist_db = MNISTDataset()
    mnist_db.create_dataset(mnist_folder, False, False, validation_size=0, seed=seed)
    test_images, test_labels = mnist_db.test

    # create neural network model
    nn_model = LenetModel()
    nn_model.create_network(nn_params)
    nn_model.initialize()

    # create workers and server
    workers = [dt_1bit.WorkerNode(nn_model) for _ in range(num_workers)]
    server = dt_1bit.AggregationNode()

    entropy = np.zeros(max_iterations)
    accuracy = np.zeros(max_iterations)

    batch_size = total_batch_size // num_workers
    for n in range(max_iterations):
        x, y = mnist_db.next_batch(total_batch_size)

        rec_rate = 0
        server.reset_node()
        for k in range(num_workers):
            # 1- get quantized gradients
            x_batch = x[k * batch_size:(k + 1) * batch_size]
            y_batch = y[k * batch_size:(k + 1) * batch_size]
            q_gW, c_gW, q_gb, c_gb = workers[k].get_quantized_gradients(x_batch, y_batch)

            # 2- compute entropy
            rec_rate += (np.sum([v.size for v in c_gW]) + np.sum(v.size for v in c_gb))
            r = np.sum([cmp.compute_entropy(v, 2) for v in q_gW]) + np.sum([cmp.compute_entropy(v, 2) for v in q_gb])
            entropy[n] += r

            # 3- aggregate gradients
            server.receive_gradient(q_gW, c_gW, q_gb, c_gb)

        # apply the gradients to the nn model
        gW, gb = server.get_aggregated_gradients()
        nn_model.apply_gradients(gW, gb)  # since they all use the same underlying nn model, no need to apply for all

        accuracy[n] = nn_model.accuracy(test_images, test_labels)
        if n % 50 == 0:
            print('{0:03d}: learning rate={1:.4f}, accuracy={2:.2f}'.format(n, nn_model.learning_rate(), accuracy[n] * 100))

    wf, bf = nn_model.get_weights()

    # computing raw rates
    r = np.sum([v.size for v in wf]) + np.sum(v.size for v in bf)  # number of parameters, represented by 1 bit
    raw_rate = r * num_workers + 32 * rec_rate
    entropy = entropy + 32 * rec_rate

    return accuracy, entropy, raw_rate, wf, bf


def evaluate_qsg(num_workers, nn_params, qsg_params, seed):
    # create database
    mnist_db = MNISTDataset()
    mnist_db.create_dataset(mnist_folder, False, False, validation_size=0, seed=seed)
    test_images, test_labels = mnist_db.test

    # create neural network model
    nn_model = LenetModel()
    nn_model.create_network(nn_params)
    nn_model.initialize()

    # create workers and server
    num_levels = qsg_params.get('num-levels', 1)
    bucket_size = qsg_params.get('bucket-size', None)
    workers = [dt_qsg.WorkerNode(nn_model) for _ in range(num_workers)]
    server = dt_qsg.AggregationNode()

    for w_id in range(num_workers):
        workers[w_id].set_quantizer(bucket_size, num_levels)

    server.set_quantizer(bucket_size, num_levels)
    avg_bits = np.log2(2 * num_levels + 1)

    entropy = np.zeros(max_iterations)
    accuracy = np.zeros(max_iterations)

    batch_size = total_batch_size // num_workers
    for n in range(max_iterations):
        x, y = mnist_db.next_batch(total_batch_size)

        rec_rate = 0
        server.reset_node()
        for k in range(num_workers):
            # 1- get quantized gradients
            x_batch = x[k * batch_size:(k + 1) * batch_size]
            y_batch = y[k * batch_size:(k + 1) * batch_size]
            qw, sw, qb, sb = workers[k].get_quantized_gradients(x_batch, y_batch)

            # 2- compute entropy
            rec_rate += (np.sum([v.size for v in sw]) + np.sum(v.size for v in sb))  # the reconstruction points
            r = np.sum([cmp.compute_entropy(v) for v in qw]) + np.sum([cmp.compute_entropy(v) for v in qb])
            entropy[n] += r

            # 3- aggregate gradients
            server.receive_gradient(qw, sw, qb, sb)

        # apply the gradients to the nn model
        gW, gb = server.get_aggregated_gradients()
        nn_model.apply_gradients(gW, gb)  # since they all use the same underlying nn model, no need to apply for all

        accuracy[n] = nn_model.accuracy(test_images, test_labels)
        if n % 50 == 0:
            print('{0:03d}: learning rate={1:.4f}, accuracy={2:.2f}'.format(n, nn_model.learning_rate(), accuracy[n] * 100))

    wf, bf = nn_model.get_weights()

    # computing raw rates
    r = np.sum([v.size for v in wf]) + np.sum(v.size for v in bf)  # number of parameters
    raw_rate = r * avg_bits * num_workers + rec_rate * 32
    entropy = entropy + rec_rate * 32

    return accuracy, entropy, raw_rate, wf, bf


def evaluate_ndqsg(num_workers, nn_params, ndqsg_params, seed):
    # create database
    mnist_db = MNISTDataset()
    mnist_db.create_dataset(mnist_folder, False, False, validation_size=0, seed=seed)
    test_images, test_labels = mnist_db.test

    # create neural network model
    nn_model = LenetModel()
    nn_model.create_network(nn_params)
    nn_model.initialize()

    # create workers and server
    ratio = ndqsg_params.get('ratio', 0.5)
    clip_thr = ndqsg_params.get('gradient-clip', None)
    num_levels = ndqsg_params.get('num-levels', ((3), (3, 1)))
    bucket_size = ndqsg_params.get('bucket-size', None)
    workers = [dt_ndqsg.WorkerNode(nn_model) for _ in range(num_workers)]
    server = dt_ndqsg.AggregationNode(num_workers)
    alphabet_size = np.zeros(num_workers)  # alphabet size of the quantized gradients
    for w_id in range(num_workers):
        dt_seed = np.random.randint(dt_ndqsg.min_seed, dt_ndqsg.max_seed)
        if w_id < (num_workers * ratio):
            q_levels = num_levels[0]
            alphabet_size[w_id] = 2 * q_levels + 1
        else:
            q_levels = num_levels[1]
            rho = q_levels[0] // q_levels[1]
            alphabet_size[w_id] = 2 * (rho // 2) + 1

        workers[w_id].set_quantizer(dt_seed, clip_thr, bucket_size, q_levels, alpha=1.0)
        server.set_quantizer(w_id, dt_seed, bucket_size, q_levels, alpha=1.0)

    avg_bits = np.mean(np.log2(alphabet_size))

    entropy = np.zeros(max_iterations)
    accuracy = np.zeros(max_iterations)

    batch_size = total_batch_size // num_workers
    for n in range(max_iterations):
        x, y = mnist_db.next_batch(total_batch_size)

        rec_rate = 0
        server.reset_node()
        for k in range(num_workers):
            # 1- get quantized gradients
            x_batch = x[k * batch_size:(k + 1) * batch_size]
            y_batch = y[k * batch_size:(k + 1) * batch_size]
            qw, sw, qb, sb = workers[k].get_quantized_gradients(x_batch, y_batch)

            # 2- aggregate gradients
            server.receive_gradient(k, qw, sw, qb, sb)

            # 3- compute entropy
            rec_rate += (np.sum([v.size for v in sw]) + np.sum(v.size for v in sb))  # the reconstruction points

            r = np.sum([cmp.compute_entropy(v) for v in qw]) + np.sum([cmp.compute_entropy(v) for v in qb])
            entropy[n] += r

        gW, gb = server.get_aggregated_gradients()
        nn_model.apply_gradients(gW, gb)

        accuracy[n] = nn_model.accuracy(test_images, test_labels)
        if n % 50 == 0:
            print('{0:03d}: learning rate={1:.4f}, accuracy={2:.2f}'.format(n, nn_model.learning_rate(), accuracy[n] * 100))

    wf, bf = nn_model.get_weights()

    # computing raw rates
    r = np.sum([v.size for v in wf]) + np.sum(v.size for v in bf)  # number of parameters
    raw_rate = r * avg_bits * num_workers + rec_rate * 32
    entropy = entropy + rec_rate * 32

    return accuracy, entropy, raw_rate, wf, bf


def test():
    bucket_size = [[400, 400, 512, 512], [32, 64, 512, 10]]

    nn_params = {'training_alg': training_algorithm,
                 'learning_rate': learning_rate,
                 'decay_rate': decay_rate,
                 'decay_step': decay_step,
                 'compute_gradients': True,
                 }

    qsg_params = {'num-levels': 3, 'bucket-size': bucket_size}
    ndqsg_params = {'ratio': 0.5, 'gradient-clip': None, 'num-levels': ((3), (3, 1)), 'bucket-size': bucket_size}

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
            np.savez(file_name, *(w0 + b0 + [seed]))

            file_name = result_folder + 'base_model.mat'
            sio.savemat(file_name, mdict={'acc': base_accuracy, 'rate': base_rate, 'w': w_base, 'b': b_base})
        else:
            # load initial weights
            file_name = result_folder + 'initial_weights.npz'
            data = np.load(file_name)
            keys = np.sort(list(data.keys()))
            num_layers = len(keys) // 2
            nn_params['initial_w'] = [data[keys[k]] for k in range(num_layers)]
            nn_params['initial_b'] = [data[keys[k]] for k in range(num_layers, 2 * num_layers)]
            seed = data[keys[-1]]

        for number_workers in [2, 4, 8, 16]:
            print('=' * 40)
            print('Running for {0} workers'.format(number_workers))
            result_folder = output_folder.format(expr) + '{}/'.format(number_workers)
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)

            print('testing 1-bit quantization method...')
            accuracy, entropy, raw_rate, w, b = evaluate_1bit(number_workers, nn_params, seed)
            file_name = result_folder + 'one_bit.mat'
            sio.savemat(file_name, mdict={'acc': accuracy, 'rate': raw_rate, 'H': entropy, 'w': w, 'b': b})

            print('testing QSG method with 1 level, no bucket...')
            qsg_params = {'num-levels': 1, 'bucket-size': None}
            accuracy, entropy, raw_rate, w, b = evaluate_qsg(number_workers, nn_params, qsg_params, seed)
            file_name = result_folder + 'qsg_nb_1.mat'
            sio.savemat(file_name, mdict={'acc': accuracy, 'rate': raw_rate, 'H': entropy, 'w': w, 'b': b})

            print('testing QSG method with 3 levels, no bucket...')
            qsg_params = {'num-levels': 3, 'bucket-size': None}
            accuracy, entropy, raw_rate, w, b = evaluate_qsg(number_workers, nn_params, qsg_params, seed)
            file_name = result_folder + 'qsg_nb_3.mat'
            sio.savemat(file_name, mdict={'acc': accuracy, 'rate': raw_rate, 'H': entropy, 'w': w, 'b': b})
            
            print('testing QSG method with 3 levels, and bucket...')
            qsg_params = {'num-levels': 3, 'bucket-size': bucket_size}
            accuracy, entropy, raw_rate, w, b = evaluate_qsg(number_workers, nn_params, qsg_params, seed)
            file_name = result_folder + 'qsg_wb_3.mat'
            sio.savemat(file_name, mdict={'acc': accuracy, 'rate': raw_rate, 'H': entropy, 'w': w, 'b': b})
            
            print('testing nested dithered quantized SG method with (3, 1) level, no bucket...')
            ndqsg_params = {'ratio': 0.5, 'gradient-clip': None, 'num-levels': ((3), (3, 1)), 'bucket-size': None}
            accuracy, entropy, raw_rate, w, b = evaluate_ndqsg(number_workers, nn_params, ndqsg_params, seed)
            file_name = result_folder + 'ndqsg_nb_3,1.mat'
            sio.savemat(file_name, mdict={'acc': accuracy, 'rate': raw_rate, 'H': entropy, 'w': w, 'b': b})

            print('testing nested dithered quantized SG method with (3, 1) level, with bucket...')
            ndqsg_params = {'ratio': 0.5, 'gradient-clip': None, 'num-levels': ((3), (3, 1)), 'bucket-size': bucket_size}
            accuracy, entropy, raw_rate, w, b = evaluate_ndqsg(number_workers, nn_params, ndqsg_params, seed)
            file_name = result_folder + 'ndqsg_wb_3,1.mat'
            sio.savemat(file_name, mdict={'acc': accuracy, 'rate': raw_rate, 'H': entropy, 'w': w, 'b': b})


if __name__ == '__main__':
    test()

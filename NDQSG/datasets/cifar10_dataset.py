"""
   CIFAR-10 data set.
   See http://www.cs.toronto.edu/~kriz/cifar.html.
"""

import os
import shutil
import urllib.request
import tarfile
import pickle

import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile
import numpy as np

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'

# dimensions of original CIFAR10 images
HEIGHT = 32
WIDTH = 32
DEPTH = 3


class Cifar10Dataset:
    def __init__(self, db_settings: dict = {}):
        self._num_classes = 10
        self._label_names = None
        self._source_url = db_settings.get('source-url', CIFAR_DOWNLOAD_URL)
        self._data_dir = db_settings.get('database-dir', '')
        self._one_hot = db_settings.get('one-hot', False)
        self._output_dim = db_settings.get('output-dimension', (24, 24))
        self._augment_training = db_settings.get('augment-training', True)

        self._train_images, self._train_labels = None, None
        self._test_images, self._test_labels = None, None
        self._num_samples = {'train': 0, 'test': 0}

    def _download_and_extract(self):
        # download CIFAR-10 and unzip it if not already downloaded.
        if not gfile.Exists(self._data_dir):
            gfile.MakeDirs(self._data_dir)

        filepath = os.path.join(self._data_dir, CIFAR_FILENAME)

        if not gfile.Exists(filepath):
            urllib.request.urlretrieve(self._source_url, filepath)

            with gfile.GFile(filepath) as f:
                size = f.size()
            print('Successfully downloaded', CIFAR_FILENAME, size, 'bytes.')

        tarfile.open(filepath, 'r:gz').extractall(self._data_dir)

        source = os.path.join(self._data_dir, CIFAR_LOCAL_FOLDER)
        files = os.listdir(source)
        for f in files:
            shutil.move(os.path.join(source, f), self._data_dir)

        os.rmdir(source)

    def _read_data_from_files(self, filenames):
        images = None
        labels = None
        for filename in filenames:
            with tf.gfile.Open(filename, 'rb') as f:
                data_dict = pickle.load(f, encoding='bytes')

            img = data_dict[b'data']
            img = np.reshape(img, (-1, 3, 32, 32))
            img = np.transpose(img, axes=(0, 2, 3, 1))
            lbl = np.array(data_dict[b'labels'])

            if images is None:
                images = img
                labels = lbl
            else:
                images = np.concatenate((images, img), axis=0)
                labels = np.concatenate((labels, lbl), axis=0)

        if self._one_hot:
            # convert labels to one-hot vectors
            num_labels = labels.shape[0]
            index_offset = np.arange(num_labels) * self._num_classes
            tmp = np.zeros((num_labels, self._num_classes))
            tmp.flat[index_offset + labels.ravel()] = 1
            labels = tmp

        return images, labels

    def _read_label_names(self):
        filename = os.path.join(self._data_dir, 'batches.meta')
        with tf.gfile.GFile(filename, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')

        self._label_names = data_dict[b'label_names']

    def _read_data_sets(self, subset):
        if subset == 'train':
            filenames = [os.path.join(self._data_dir, 'data_batch_%d' % i) for i in range(1, 6)]
        elif subset == 'test':
            filenames = [os.path.join(self._data_dir, 'test_batch')]
        else:
            raise ValueError('not a valid subset is provided.')

        files_exist = np.all([os.path.exists(fname) for fname in filenames])
        if not files_exist:
            self._download_and_extract()

        images, labels = self._read_data_from_files(filenames)

        return images, labels

    def _load_from_file(self, subset=('train', 'test')):
        self._read_label_names()

        if 'train' in subset:
            self._train_images, self._train_labels = self._read_data_sets('train')
            self._num_samples['train'] = self._train_labels.shape[0]

        if 'test' in subset:
            self._test_images, self._test_labels = self._read_data_sets('test')
            self._num_samples['test'] = self._test_labels.shape[0]

    def _augment_training_image(self, image, label):
        # Randomly crop a [height, width] section of the image.
        if (self._output_dim[0] >= HEIGHT) or (self._output_dim[1] >= WIDTH):
            image = tf.image.resize_with_crop_or_pad(image, self._output_dim[0] + 8, self._output_dim[1] + 8)

        image = tf.image.random_crop(image, [self._output_dim[0], self._output_dim[1], 3])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_standardization(image)

        # Set the shapes of tensors.
        image.set_shape([self._output_dim[0], self._output_dim[1], 3])

        return image, label

    def _prepare_test_image(self, image, label):
        # Crop the central [height, width] section of the image.
        image = tf.image.resize_with_crop_or_pad(image, self._output_dim[0], self._output_dim[1])

        # Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_standardization(image)

        # Set the shapes of tensors.
        image.set_shape([self._output_dim[0], self._output_dim[1], 3])

        return image, label

    def _create_training_dataset(self, batch_size, num_parallel_calls):
        assert self._train_images is not None, 'training images has not been loaded from file.'

        train_db = tf.data.Dataset.from_tensor_slices((self._train_images, self._train_labels))
        train_db = train_db.shuffle(self._num_samples['train']).repeat()

        if self._augment_training:
            train_db = train_db.map(self._augment_training_image, num_parallel_calls=num_parallel_calls)
        else:
            train_db = train_db.map(self._prepare_test_image, num_parallel_calls=num_parallel_calls)

        train_db = train_db.batch(batch_size, drop_remainder=True)
        # prefetch data
        train_db = train_db.prefetch(1)

        return train_db

    def _create_test_dataset(self, num_parallel_calls):
        assert self._test_images is not None, 'test images has not been loaded from file.'

        test_db = tf.data.Dataset.from_tensor_slices((self._test_images, self._test_labels))
        test_db = test_db.repeat()

        test_db = test_db.map(self._prepare_test_image, num_parallel_calls=num_parallel_calls)
        test_db = test_db.batch(batch_size=self._num_samples['test'])

        return test_db

    def create_dataset(self, subset={'train', 'test'}, batch_size=1, num_parallel_calls=16):
        databases = {}
        self._load_from_file(subset)

        if 'train' in subset:
            databases['train'] = self._create_training_dataset(batch_size, num_parallel_calls)

        if 'test' in subset:
            databases['test'] = self._create_test_dataset(num_parallel_calls)

        assert databases != {}, 'no valid subset of CIFAR10 database is provided.'

        # get the types and shapes of the database outputs
        db_types = next(iter(databases.items()))[1].output_types
        db_shapes = next(iter(databases.items()))[1].output_shapes

        # define the iterator and generate different initializers
        iterator = tf.data.Iterator.from_structure(db_types, db_shapes)
        
        initializer_op = {}
        for k in databases.keys():
            # define different initializers
            initializer_op[k] = iterator.make_initializer(databases[k])
        
        image_batch, label_batch = iterator.get_next(name='input-data')

        return image_batch, label_batch, initializer_op

    def get_number_samples(self, subset='train'):
        assert subset in ['train', 'test'], 'no valid subset of dataset is provided!'
        return self._num_samples[subset]

    def get_label_texts(self):
        return self._label_names

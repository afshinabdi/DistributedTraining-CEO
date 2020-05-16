"""
   implementation of the MNIST/Fashion MNIST database
"""
import os
import numpy as np
import gzip
import urllib.request

import tensorflow as tf
from tensorflow.python.platform import gfile    # pylint: disable=E0611

DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
# For Fashion MNIST, use the following link: 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'


def _check_download_file(filename, dir_name, source_url):
    """Download the data from source url, unless it's already here.

    Args:
        filename: string, name of the file in the directory.
        dir_name: string, path to working directory.
        source_url: url to download from if file doesn't exist.

    Returns:
        Path to resulting file.
    """
    if not gfile.Exists(dir_name):
        gfile.MakeDirs(dir_name)
    filepath = os.path.join(dir_name, filename)

    if not gfile.Exists(filepath):
        urllib.request.urlretrieve(source_url, filepath)

        with gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')

    return filepath


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def _dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


def _extract_images(f):
    """Extract the images into a 4D uint8 np array [index, y, x, depth].

    Args:
      f: A file object that can be passed into a gzip reader.

    Returns:
      data: A 4D uint8 np array [index, y, x, depth].

    Raises:
      ValueError: If the bytestream does not start with 2051.

    """
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))

        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        images = np.frombuffer(buf, dtype=np.uint8)
        images = images.reshape(num_images, rows, cols, 1)

        images = images.astype(np.float32) / 255
        return images


def _extract_labels(f):
    """Extract the labels into a 1D uint8 numpy array [index].

    Args:
      f: A file object that can be passed into a gzip reader.
      one_hot: Does one hot encoding for the result.
      num_classes: Number of classes for the one hot encoding.

    Returns:
      labels: a 1D uint8 numpy array.

    Raises:
      ValueError: If the bystream doesn't start with 2049.
    """
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        assert magic == 2049, 'Invalid magic number %d in MNIST label file: %s' % (magic, f.name)

        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)

    return labels.astype(np.int32)


def _read_data_sets(db_dir, source_url):

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    local_file = _check_download_file(TRAIN_IMAGES, db_dir, source_url + TRAIN_IMAGES)
    with gfile.Open(local_file, 'rb') as f:
        train_images = _extract_images(f)

    local_file = _check_download_file(TRAIN_LABELS, db_dir, source_url + TRAIN_LABELS)
    with gfile.Open(local_file, 'rb') as f:
        train_labels = _extract_labels(f)

    local_file = _check_download_file(TEST_IMAGES, db_dir, source_url + TEST_IMAGES)
    with gfile.Open(local_file, 'rb') as f:
        test_images = _extract_images(f)

    local_file = _check_download_file(TEST_LABELS, db_dir, source_url + TEST_LABELS)
    with gfile.Open(local_file, 'rb') as f:
        test_labels = _extract_labels(f)

    return train_images, train_labels, test_images, test_labels


def create_dataset(
    db_dir,
    batch_size,
    vector_fmt=False,
    one_hot=True,
    validation_size=5000,
    source_url=DEFAULT_SOURCE_URL,
    dataset_fmt=False
):
    """
       main function to create the mnist data set
        Args:
            db_dir: string, disrectory of the files of the mnist database.
            batch_size: integer or placeholder,  training batch size.
            vector_fmt: the datapoints are in the vectorized (-1, 784) or image (-1, 28, 28, 1) format.
            one_hot: boolean, labels are one-hot represented or not.
            validation_size: integer, number of samples in the validation set.
            source_url: url to download from if file doesn't exist.

        Returns:
            images, labels
            the initializer operators for the datasets
            number of samples in each subset of the database
    """
    # read data from files
    train_images, train_labels, test_images, test_labels = _read_data_sets(db_dir, source_url)

    if one_hot:
        train_labels = _dense_to_one_hot(train_labels, num_classes=10)
        test_labels = _dense_to_one_hot(test_labels, num_classes=10)

    if vector_fmt:
        train_images = np.reshape(train_images, newshape=(-1, 784))
        test_images = np.reshape(test_images, newshape=(-1, 784))

    # separate the validation data
    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'.format(len(train_images), validation_size)
        )

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    number_samples = {'train': len(train_labels), 'validation': len(validation_labels), 'test': len(test_labels)}

    # create training dataset
    train_db = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_db = train_db.shuffle(number_samples['train']).repeat()
    train_db = train_db.batch(batch_size)
    # prefetch data
    train_db = train_db.prefetch(1)

    # create validation dataset
    valid_db = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
    valid_db = valid_db.batch(number_samples['validation'])

    # create test dataset
    test_db = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_db = test_db.batch(number_samples['test'])

    if dataset_fmt:
        return train_db, valid_db, test_db, number_samples

    # define the iterator and different initializers
    iterator = tf.data.Iterator.from_structure(train_db.output_types, train_db.output_shapes)

    images, labels = iterator.get_next()

    train_init_op = iterator.make_initializer(train_db)
    valid_init_op = iterator.make_initializer(valid_db)
    test_init_op = iterator.make_initializer(test_db)

    init_op = {'train': train_init_op, 'validation': valid_init_op, 'test': test_init_op}

    return images, labels, init_op, number_samples


class MNISTDataset:
    def __init__(self):
        self._train_images = None
        self._train_labels = None
        self._validation_images = None
        self._validation_labels = None
        self._test_images = None
        self._test_labels = None
        self._number_samples = {'train': 0, 'validation': 0, 'test': 0}

        self._seed = None
        self._index_pos = 0
        self._shuffled_index = [0]

    @property
    def number_samples(self):
        return self._number_samples

    @property
    def train_data(self):
        return self._train_images, self._train_labels

    @property
    def validation(self):
        return self._validation_images, self._validation_labels

    @property
    def test_data(self):
        return self._test_images, self._test_labels

    def _prepare_samples(self, db_dir, vector_fmt, one_hot, validation_size):
        # read data from files
        self._train_images, self._train_labels, self._test_images, self._test_labels = _read_data_sets(
            db_dir, source_url=DEFAULT_SOURCE_URL
        )

        if one_hot:
            self._train_labels = _dense_to_one_hot(self._train_labels, num_classes=10)
            self._test_labels = _dense_to_one_hot(self._test_labels, num_classes=10)

        if vector_fmt:
            self._train_images = np.reshape(self._train_images, newshape=(-1, 784))
            self._test_images = np.reshape(self._test_images, newshape=(-1, 784))

        # separate the validation data
        if not 0 <= validation_size <= len(self._train_images):
            raise ValueError(
                'Validation size should be between 0 and {}. Received: {}.'.format(
                    len(self._train_images), validation_size
                )
            )

        self._validation_images = self._train_images[:validation_size]
        self._validation_labels = self._train_labels[:validation_size]
        self._train_images = self._train_images[validation_size:]
        self._train_labels = self._train_labels[validation_size:]

        self._number_samples = {
            'train': len(self._train_labels),
            'validation': len(self._validation_labels),
            'test': len(self._test_labels)
        }

    def _reset_shuffled_index(self):
        np.random.seed(self._seed)

        self._shuffled_index = np.arange(0, self._number_samples['train'])
        np.random.shuffle(self._shuffled_index)
        self._index_pos = 0
        # update seed for reproducibility and avoiding conflicts with other rand calls
        self._seed = np.random.randint(1000, 1000000)

    def create_dataset(self, db_dir, vector_fmt=False, one_hot=True, validation_size=5000, seed=None):
        self._seed = np.random.randint(1000, 1000000) if (seed is None) else seed

        # read database samples from file or download them if necessary
        self._prepare_samples(db_dir, vector_fmt, one_hot, validation_size)
        self._reset_shuffled_index()

    def next_batch(self, batch_size):
        if (self._index_pos + batch_size) >= self._number_samples['train']:
            self._reset_shuffled_index()

        index = self._shuffled_index[self._index_pos:(self._index_pos + batch_size)]
        self._index_pos += batch_size

        return self._train_images[index], self._train_labels[index]

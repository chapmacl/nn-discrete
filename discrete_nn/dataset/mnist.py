"""
Contains the MNIST dataset
"""

from discrete_nn import settings
import os
import shutil
from urllib import request
import gzip
import struct
import numpy as np
from torch.utils.data import Dataset
import torch


class DatasetMNIST(Dataset):
    """
    Dataset for pytorch's DataLoader
    """

    def __init__(self, x, y, device, feature_shape):
        """
        Container for a mnist dataset
        :param x: the features
        :param y: the labels
        :param device: the device to store the tensors in
        :param feature_shape: the shape that data should be returned in:
            * flat: each data instance is a row vector
            * 2d: shape used for networks with 2d input. Each minibatch is of the shape ((batch_size, 1, 28, 28))

        """
        self.x = torch.from_numpy(x) * 2 - 1
        self.y = torch.from_numpy(y).long()
        if feature_shape == "2d":
            self.x = self.x.reshape((self.x.shape[0], 1, 28, 28))
        elif feature_shape != "flat":
            # invalid feature_shape
            raise ValueError(f"invalid feature_shape {feature_shape}")
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item_inx):
        return self.x[item_inx], self.y[item_inx]


class MNIST:
    @staticmethod
    def _download_and_uncompress(url, dest_filepath, replace=False):
        if os.path.isfile(dest_filepath) and not replace:
            return
        # makes sure the directory exists
        dir_path = os.path.dirname(dest_filepath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        temp_file = os.path.join(os.path.dirname(dest_filepath), "temp.gz")
        request.urlretrieve(url, temp_file)
        with gzip.open(temp_file, 'rb') as f_in, open(dest_filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(temp_file)

    @staticmethod
    def _load_target_file(filepath, magic_number, number_images):
        with open(filepath, 'rb') as f:
            magic_read, n_images_read = struct.unpack('>II', f.read(8))
            if magic_read != magic_number:
                raise ValueError(f"magic number read from file ({magic_read}) differs from provided ({magic_number})")
            if n_images_read != number_images:
                raise ValueError(f"number of images from file header ({n_images_read}) "
                                 f"differs from provided ({number_images})")

            return np.fromfile(f, dtype=np.int8)

    @staticmethod
    def _load_input_file(filepath, magic_number, number_images, image_x, image_y):
        with open(filepath, 'rb') as f:
            magic_read, n_images_read, n_rows, n_cols = struct.unpack('>IIII', f.read(16))
            if magic_read != magic_number:
                raise ValueError(f"magic number read from file ({magic_read}) differs from provided ({magic_number})")
            if n_images_read != number_images:
                raise ValueError(f"number of images from file header ({n_images_read}) "
                                 f"differs from provided ({number_images})")
            if image_x != n_cols:
                raise ValueError(f"x from file ({n_cols}) is not ({image_x})")

            if image_y != n_rows:
                raise ValueError(f"x from file ({n_rows}) is not ({image_y})")

            return np.fromfile(f, dtype=np.uint8).reshape(n_images_read, n_rows * n_cols).astype(np.float32) / 255.

    def __init__(self, device, feature_shape, force_download=False):
        """
        Initializes a MNIST dataset holder
        :param device: the device to stores the tensors in
        :param feature_shape:the shape that data should be returned in:
            * flat: each data instance is a row vector
            * 2d: shape used for networks with 2d input. Each minibatch is of the shape ((batch_size, 1, 28, 28))
        :param force_download: if true will replace local cache
        """
        dataset_folder = os.path.join(settings.dataset_path, "mnist")
        url_x_train = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        x_train_path = os.path.join(dataset_folder, "mnist_x_train.bin")
        self._download_and_uncompress(url_x_train, x_train_path, replace=force_download)
        url_y_train = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
        y_train_path = os.path.join(dataset_folder, "mnist_y_train.bin")
        self._download_and_uncompress(url_y_train, y_train_path, replace=force_download)
        url_x_test = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
        x_test_path = os.path.join(dataset_folder, "mnist_x_test.bin")
        self._download_and_uncompress(url_x_test, x_test_path, replace=force_download)
        url_y_test = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
        y_test_path = os.path.join(dataset_folder, "mnist_y_test.bin")
        self._download_and_uncompress(url_y_test, y_test_path, replace=force_download)

        x_train = self._load_input_file(x_train_path, 2051, 60000, 28, 28)
        y_train = self._load_target_file(y_train_path, 2049, 60000)

        # need to split training set into training and validation set
        x_val: np.ndarray = x_train[50000:]
        y_val: np.ndarray = y_train[50000:]
        x_train: np.ndarray = x_train[:50000]
        y_train: np.ndarray = y_train[:50000]

        self.train = DatasetMNIST(x_train, y_train, device, feature_shape)
        self.validation = DatasetMNIST(x_val, y_val, device, feature_shape)

        x_test: np.ndarray = self._load_input_file(x_test_path, 2051, 10000, 28, 28)
        y_test: np.ndarray = self._load_target_file(y_test_path, 2049, 10000)
        self.test = DatasetMNIST(x_test, y_test, device, feature_shape)

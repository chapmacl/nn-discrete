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
from torch.utils.data import DataLoader, Dataset


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

    def __init__(self, force_download=False):
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

        self.x_train = self._load_input_file(x_train_path, 2051, 60000, 28, 28)
        self.y_train = self._load_target_file(y_train_path, 2049, 60000)

        # need to split training set into training and validation set
        self.x_val: np.ndarray = self.x_train[50000:]
        self.y_val: np.ndarray = self.y_train[50000:]
        self.x_train: np.ndarray = self.x_train[:50000]
        self.y_train: np.ndarray = self.y_train[:50000]

        self.x_test: np.ndarray = self._load_input_file(x_test_path, 2051, 10000, 28, 28)
        self.y_test: np.ndarray = self._load_target_file(y_test_path, 2049, 10000)
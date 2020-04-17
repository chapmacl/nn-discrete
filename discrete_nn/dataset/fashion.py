from typing import Tuple
import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from discrete_nn.settings import dataset_path

def fashion_mnist_dataloaders(batch_size, device, feature_format) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the Fashion MNIST dataset,
    :param batch_size:
    :param device:
    :param feature_format: 2d or flat
    :return: (train, validation, test) data loaders
    """

    # creates the dataloader for pytorch
    mnist_fashion_path = os.path.join(dataset_path, "fashion")

    feature_fn = partial(_transform_feature, feature_format=feature_format, device=device)
    target_fn = partial(_transform_target, device=device)

    train_val_dataset = FashionMNIST(mnist_fashion_path, download=True, train=True, transform=feature_fn,
                                     target_transform=target_fn)

    train_size = int(len(train_val_dataset) * 0.8)
    eval_size = len(train_val_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, eval_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    test_loader = DataLoader(FashionMNIST(mnist_fashion_path, download=True, train=False, transform=feature_fn,
                                          target_transform=target_fn),
                             batch_size=batch_size)
    return train_loader, validation_loader, test_loader


# these methods have to stay here bc they need to be serializable by torch

def _transform_target(target, device):
    return torch.tensor(target).to(device)


_to_tensor_method = ToTensor()


def _transform_feature(pil_image, feature_format, device):
    if feature_format == "2d":
        return _to_tensor_method(pil_image).to(device)
    elif feature_format == "flat":
        return _to_tensor_method(pil_image).reshape(-1).to(device)
    else:
        raise ValueError(f"invalid value for feature format:{feature_format}")

"""
This module implements the real valued (non convolutionary) network for the conv dataset
"""
import os

import torch
from torch.utils.data import DataLoader

from discrete_nn.dataset.mnist import MNIST
from discrete_nn.settings import model_path
from discrete_nn.layers.type_defs import DiscreteWeights
from discrete_nn.models.conv.logit_sign import ConvLogitSign


def train_model(real_model_folder):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = 100
    # basic dataset holder
    mnist = MNIST(device, "2d")
    # creates the dataloader for pytorch
    train_loader = DataLoader(dataset=mnist.train, batch_size=batch_size,
                              shuffle=True)
    validation_loader = DataLoader(dataset=mnist.validation, batch_size=batch_size,
                                   shuffle=False)
    test_loader = DataLoader(dataset=mnist.test, batch_size=batch_size,
                             shuffle=False)

    real_model_param_path = os.path.join(model_path, real_model_folder,
                                         "ConvReal.param.pickle")

    real_param = torch.load(real_model_param_path, map_location="cpu")
    logit_net = ConvLogitSign(real_param, DiscreteWeights.TERNARY)
    logit_net = logit_net.to(device)

    # evaluate first logit model before training, train and evaluate again

    logit_net.train_model(train_loader, validation_loader, test_loader, 200, "MNIST-Conv-Ternary-Sign", True)


if __name__ == "__main__":
    train_model("MNIST-pi-real-trained-2020-3-8--h17m34")

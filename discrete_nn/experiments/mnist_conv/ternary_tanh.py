from torch.utils.data import DataLoader
import torch

import os

from discrete_nn.dataset.mnist import MNIST
from discrete_nn.settings import model_path
from discrete_nn.models.conv.ternary_tanh import ConvLogitTanh
from discrete_nn.layers.type_defs import DiscreteWeights


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

    real_param = torch.load(real_model_param_path)
    logit_net = ConvLogitTanh(real_param, DiscreteWeights.TERNARY)
    logit_net = logit_net.to(device)

    # evaluate first logit model before training, train and evaluate again
    logit_net.train_model(train_loader, validation_loader, test_loader, 200, "MNIST-Conv-Ternary-Tanh", True)


if __name__ == "__main__":
    train_model("MNIST-real-conv-trained-2020-3-4--h11m53")

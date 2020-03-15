"""
This module implements the real valued (non convolutionary) network for the mnist dataset
"""
import torch
from torch.utils.data import DataLoader

import os
import pickle

from discrete_nn.dataset.mnist import MNIST
from discrete_nn.settings import model_path
from discrete_nn.layers.type_defs import ValueTypes, DiscreteWeights, InputFormat
from discrete_nn.layers.logit_linear import TernaryLinear
from discrete_nn.layers.conv import LogitConv
from discrete_nn.layers.sign import DistributionSign
from discrete_nn.layers.pool import DistributionMaxPool
from discrete_nn.layers.distribution_batchnorm import DistributionBatchnorm
from discrete_nn.layers.local_reparametrization import LocalReparametrization
from discrete_nn.models.mnist.real import MnistReal
from discrete_nn.models.base_model import LogitModel
from discrete_nn.models.evaluation_utils import evaluate_discretized_from_logit_models

from discrete_nn.layers.Flatten import Flatten


class MnistTernarySign(LogitModel):
    """
    Real valued (non convolutionary) network for the mnist dataset
    """

    def __init__(self, real_model_params):
        """

        :param real_model_params: a dictionary containing the real weights of the pretrained model
        """
        super().__init__()
        self.netlayers = torch.nn.Sequential(
            LogitConv(1, ValueTypes.REAL, 32, 5, 1, real_model_params["L1_Conv_W"], None, DiscreteWeights.TERNARY),
            # torch.nn.Conv2d(1, 32, 5, stride=1, bias=False),
            DistributionMaxPool(2),  # torch.nn.MaxPool2d(2),
            DistributionBatchnorm(InputFormat.FEATURE_MAP, 32, real_model_params["L1_BatchNorm_W"],
                                  real_model_params["L1_BatchNorm_b"]),  # torch.nn.BatchNorm2d(32, momentum=0.1),
            DistributionSign(InputFormat.FEATURE_MAP),
            #
            LocalReparametrization(),
            torch.nn.Dropout(p=0.2),

            # torch.nn.Conv2d(32, 64, 5, stride=1, bias=False),
            LogitConv(32, ValueTypes.REAL, 64, 5, 1, real_model_params["L2_Conv_W"], None, DiscreteWeights.TERNARY),
            DistributionMaxPool(2),  # torch.nn.MaxPool2d(2),
            DistributionBatchnorm(InputFormat.FEATURE_MAP, 64, real_model_params["L2_BatchNorm_W"],
                                  real_model_params["L2_BatchNorm_b"]),
            DistributionSign(InputFormat.FEATURE_MAP),
            LocalReparametrization(),
            #
            Flatten(),
            torch.nn.Dropout(p=0.3),
            TernaryLinear(3136, ValueTypes.REAL, 512, real_model_params["L3_Linear_W"],
                          real_model_params["L3_Linear_b"], normalize_activations=False),
            DistributionBatchnorm(InputFormat.FLAT_ARRAY, 512, real_model_params["L3_BatchNorm_W"],
                                  real_model_params["L3_BatchNorm_b"]),  # torch.nn.BatchNorm1d(512, momentum=0.1),
            DistributionSign(InputFormat.FLAT_ARRAY),
            LocalReparametrization(),
            #
            TernaryLinear(512, ValueTypes.REAL, 10, real_model_params["L4_Linear_W"], real_model_params["L4_Linear_b"],
                          normalize_activations=True),
            LocalReparametrization(),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-10)
        self.loss_funct = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # takes image vector
        return self.netlayers(x)

    def get_net_parameters(self):
        return self.state_dict()

    def set_net_parameters(self, param_dict):
        for k, v in param_dict.items():
            self.state_dict()[k][:] = v

    def generate_discrete_networks(self, method: str) -> MnistReal:
        """

        :param method: sample or argmax
        :return:
        """
        raise NotImplementedError


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
                                         "MnistReal.param.pickle")
    with open(real_model_param_path, "rb") as f:
        real_param = pickle.load(f)
        logit_net = MnistTernarySign(real_param)
    logit_net = logit_net.to(device)

    # evaluate first logit model before training, train and evaluate again

    logit_net.train_model(train_loader, validation_loader, test_loader, 100, "MNIST-Conv-Sign-Ternary", True)


if __name__ == "__main__":
    train_model("MNIST-real-conv-trained-2020-3-4--h11m53")
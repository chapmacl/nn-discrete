"""
This module implements the real valued (non convolutionary) network for the mnist dataset
"""
import os
import pickle

import torch
from torch.utils.data import DataLoader

from discrete_nn.dataset.mnist import MNIST
from discrete_nn.settings import model_path
from discrete_nn.layers.type_defs import ValueTypes, InputFormat
from discrete_nn.layers.logit_linear import TernaryLinear
from discrete_nn.layers.sign import DistributionSign

from discrete_nn.models.mnist_pi.discrete_sign import MnistPiDiscreteSign
from discrete_nn.layers.distribution_batchnorm import DistributionBatchnorm
from discrete_nn.layers.local_reparametrization import LocalReparametrization
from discrete_nn.models.mnist_pi.real import MnistPiReal
from discrete_nn.models.base_model import BaseModel
from discrete_nn.models.evaluation_utils import evaluate_discretized_from_logit_models


class MnistPiTernarySign(BaseModel):
    """
    Real valued (non convolutionary) network for the mnist dataset
    """

    def __init__(self, real_model_params):
        """

        :param real_model_params: a dictionary containing the real weights of the pretrained model
        """
        super().__init__()


        # defining all the network's layers
        self.netlayers = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            TernaryLinear(784, ValueTypes.REAL, 1200, real_model_params["L1_Linear_W"],
                          real_model_params["L1_Linear_b"]),
            DistributionBatchnorm(InputFormat.FLAT_ARRAY, 1200, real_model_params["L1_BatchNorm_W"],
                                  real_model_params["L1_BatchNorm_b"]),  # ?, momentum=0.1)
            DistributionSign(InputFormat.FLAT_ARRAY),
            LocalReparametrization(),  # outputs a value and not a dist.

            torch.nn.Dropout(p=0.2),
            TernaryLinear(1200, ValueTypes.REAL, 1200, real_model_params["L2_Linear_W"],
                          real_model_params["L2_Linear_b"]),
            DistributionBatchnorm(InputFormat.FLAT_ARRAY, 1200, real_model_params["L2_BatchNorm_W"],
                                  real_model_params["L2_BatchNorm_b"]),  # ?, momentum=0.1)
            DistributionSign(InputFormat.FLAT_ARRAY),
            LocalReparametrization(),  # outputs a value and not a dist.

            torch.nn.Dropout(p=0.3),
            TernaryLinear(1200, ValueTypes.REAL, 10, real_model_params["L3_Linear_W"],
                          real_model_params["L3_Linear_b"], normalize_activations=True),
            LocalReparametrization())  # last layer outputs the unnormalized loglikelihood used by the softmax later

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-10)
        print(self.optimizer)
        self.loss_funct = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # takes image vector
        return self.netlayers(x)

    def get_net_parameters(self):
        return self.state_dict()

    def set_net_parameters(self, param_dict):
        self.load_state_dict(param_dict, strict=False)

    def generate_discrete_networks(self, method: str) -> MnistPiReal:
        """

        :param method: sample or argmax
        :return:
        """
        # state dicts
        l1_layer: TernaryLinear = self.netlayers[1]
        l1_sampled_w, l1_sampled_b = l1_layer.generate_discrete_network(method)
        l2_layer: TernaryLinear = self.netlayers[6]
        l2_sampled_w, l2_sampled_b = l2_layer.generate_discrete_network(method)
        l3_layer: TernaryLinear = self.netlayers[11]
        l3_sampled_w, l3_sampled_b = l3_layer.generate_discrete_network(method)
        state_dict = {
            "L1_Linear_W": l1_sampled_w,
            "L1_Linear_b": l1_sampled_b,
            "L1_BatchNorm_W": self.state_dict()['netlayers.2.gamma'],
            "L1_BatchNorm_b": self.state_dict()['netlayers.2.beta'],
            "L2_Linear_W": l2_sampled_w,
            "L2_Linear_b": l2_sampled_b,
            "L2_BatchNorm_W": self.state_dict()['netlayers.7.gamma'],
            "L2_BatchNorm_b": self.state_dict()['netlayers.7.beta'],
            "L3_Linear_W": l3_sampled_w,
            "L3_Linear_b": l3_sampled_b
        }

        real_net = MnistPiDiscreteSign()
        real_net.set_net_parameters(state_dict)
        return real_net


def train_model(real_model_folder):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = 100
    # basic dataset holder
    mnist = MNIST(device, "flat")
    # creates the dataloader for pytorch
    train_loader = DataLoader(dataset=mnist.train, batch_size=batch_size,
                              shuffle=True)
    validation_loader = DataLoader(dataset=mnist.validation, batch_size=batch_size,
                                   shuffle=False)
    test_loader = DataLoader(dataset=mnist.test, batch_size=batch_size,
                             shuffle=False)
    """
    real_model_param_path = os.path.join(model_path, real_model_folder,
                                         "MnistPiReal.param.pickle")

    with open(real_model_param_path, "rb") as f:
        real_param = pickle.load(f)
        logit_net = MnistPiTernarySign(real_param)
    """
    # create dummy mnist pi class
    logit_net = MnistPiTernarySign(MnistPiReal().get_net_parameters())
    logit_net.set_net_parameters(torch.load("/home/tbellfelix/Downloads/MnistPiTernarySign.param.pickle", map_location="cpu"))

    logit_net = logit_net.to(device)


    # discretizing and evaluating
    evaluate_discretized_from_logit_models(logit_net, "sample", test_loader, 10,
                                           os.path.join(model_path,
                                                        "MNIST-Pi-Ternary_sing_untrained_discrete_sample"))
    evaluate_discretized_from_logit_models(logit_net, "argmax", test_loader, 1,
                                           os.path.join(model_path,
                                                        "MNIST-Pi-Ternary_sign_untrained_discrete_argmax"))

    # evaluate first logit model before training, train and evaluate again

    #logit_net.train_model(train_loader, validation_loader, test_loader, 100, "MNIST-Pi-Sign-Ternary", True)
    evaluate_discretized_from_logit_models(logit_net, "sample", test_loader, 10,
                                           os.path.join(model_path,
                                                        "MNIST-Pi-Ternary_sign_trained_discrete_sample"))
    evaluate_discretized_from_logit_models(logit_net, "argmax", test_loader, 1,
                                           os.path.join(model_path,
                                                        "MNIST-Pi-Ternary_sign_trained_discrete_argmax"))

if __name__ == "__main__":
    train_model("MNIST-pi-real-trained-2020-3-8--h17m34")

"""
This module implements the real valued (non convolutionary) network for the mnist dataset
"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import torch
import numpy as np
import tqdm
import os
import pickle
from sklearn.metrics import accuracy_score

from discrete_nn.dataset.mnist import MNIST
from discrete_nn.settings import model_path
from discrete_nn.layers.type_defs import ValueTypes, DiscreteWeights
from discrete_nn.layers.logit_linear import TernaryLinear
from discrete_nn.layers.conv import LogitConv
from discrete_nn.layers.local_reparametrization import LocalReparametrization
from discrete_nn.models.fashion_conv.real import FashionConvReal
from discrete_nn.models.base_model import BaseModel

from discrete_nn.layers.Flatten import Flatten

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class FashionConvTernaryTanh(BaseModel):
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
            LocalReparametrization(),
            torch.nn.BatchNorm2d(32, momentum=0.1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(2),
            #
            torch.nn.Dropout(p=0.2),
            # torch.nn.Conv2d(32, 64, 5, stride=1, bias=False),
            LogitConv(32, ValueTypes.REAL, 64, 5, 1, real_model_params["L2_Conv_W"], None, DiscreteWeights.TERNARY),
            LocalReparametrization(),
            torch.nn.BatchNorm2d(64, momentum=0.1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(2),

            #
            Flatten(),
            torch.nn.Dropout(p=0.3),
            TernaryLinear(3136, ValueTypes.REAL, 512, real_model_params["L3_Linear_W"],
                          real_model_params["L3_Linear_b"], normalize_activations=False),
            LocalReparametrization(),
            torch.nn.BatchNorm1d(512, momentum=0.1),
            torch.nn.Tanh(),
            #
            TernaryLinear(512, ValueTypes.REAL, 10, real_model_params["L4_Linear_W"], real_model_params["L4_Linear_b"],
                          normalize_activations=True),
            LocalReparametrization(),
        )

        self.state_dict()["netlayers.2.weight"] = real_model_params["L1_BatchNorm_W"]
        self.state_dict()["netlayers.2.bias"] = real_model_params["L1_BatchNorm_b"]
        self.state_dict()["netlayers.8.weight"] = real_model_params["L2_BatchNorm_W"]
        self.state_dict()["netlayers.8.bias"] = real_model_params["L2_BatchNorm_b"]
        self.state_dict()["netlayers.15.weight"] = real_model_params["L3_BatchNorm_W"]
        self.state_dict()["netlayers.15.bias"] = real_model_params["L3_BatchNorm_b"]

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-8)
        self.loss_funct = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # takes image vector
        return self.netlayers(x)

    def get_net_parameters(self):
        return self.state_dict()

    def set_net_parameters(self, param_dict):
        for k, v in param_dict.items():
            self.state_dict()[k][:] = v

    def generate_discrete_networks(self, method: str) -> FashionConvReal:
        """

        :param method: sample or argmax
        :return:
        """
        # state dicts
        l1_layer: LogitConv = self.netlayers[0]
        l1_sampled_w, l1_sampled_b = l1_layer.generate_discrete_network(method)
        l2_layer: LogitConv = self.netlayers[6]
        l2_sampled_w, l2_sampled_b = l2_layer.generate_discrete_network(method)
        l3_layer: TernaryLinear = self.netlayers[13]
        l3_sampled_w, l3_sampled_b = l3_layer.generate_discrete_network(method)
        l4_layer: TernaryLinear = self.netlayers[17]
        l4_sampled_w, l4_sampled_b = l4_layer.generate_discrete_network(method)
        state_dict = {
            "L1_Conv_W": l1_sampled_w,
            "L1_BatchNorm_W": self.state_dict()['netlayers.2.weight'],
            "L1_BatchNorm_b": self.state_dict()['netlayers.2.bias'],
            "L2_Conv_W": l2_sampled_w,
            "L2_BatchNorm_W": self.state_dict()['netlayers.8.weight'],
            "L2_BatchNorm_b": self.state_dict()['netlayers.8.bias'],
            "L3_Linear_W": l3_sampled_w,
            "L3_Linear_b": l3_sampled_b,
            "L3_BatchNorm_W": self.state_dict()["netlayers.15.weight"],
            "L3_BatchNorm_b": self.state_dict()["netlayers.15.bias"],
            "L4_Linear_W": l4_sampled_w,
            "L4_Linear_b": l4_sampled_b
        }
        real_net = FashionConvReal()
        real_net.to(device)
        real_net.set_net_parameters(state_dict)
        return real_net


def train_model(real_model_folder):

    batch_size = 100
    ToTensorMethod = ToTensor()

    def transform_input(pil_image):
        return ToTensorMethod(pil_image).to(device)

    def transform_target(target):
        return torch.tensor(target).to(device)

    from discrete_nn.settings import dataset_path
    mnist_fashion_path = os.path.join(dataset_path, "fashion")
    train_val_dataset = FashionMNIST(mnist_fashion_path, download=True, train=True, transform=transform_input,
                                     target_transform=transform_target)

    train_size = int(len(train_val_dataset) * 0.8)
    eval_size = len(train_val_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, eval_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    test_loader = DataLoader(FashionMNIST(mnist_fashion_path, download=True, train=False, transform=transform_input,
                                          target_transform=transform_target),
                             batch_size=batch_size)

    print('Using device:', device)

    real_model_param_path = os.path.join(model_path, real_model_folder,
                                         "FashionConvReal.param.pickle")
    with open(real_model_param_path, "rb") as f:
        real_param = pickle.load(f)
        logit_net = FashionConvTernaryTanh(real_param)
    logit_net = logit_net.to(device)
    # discretizing and evaluating
    # todo should probably generate several sampled ones?
    discrete_net = logit_net.generate_discrete_networks("sample")
    discrete_net = discrete_net.to(device)
    discrete_net.evaluate_and_save_to_disk(test_loader, "FashionMnistConv-ex3.1_untrained_discretized_ternary_sample")
    discrete_net = logit_net.generate_discrete_networks("argmax")
    discrete_net = discrete_net.to(device)
    discrete_net.evaluate_and_save_to_disk(test_loader, "FashionMnistConv-ex3.1_untrained_discretized_ternary_argmax")

    # evaluate first logit model before training, train and evaluate again
    logit_net.train_model(train_loader, validation_loader, test_loader, 100, "FashionMnistConvTernary", True)

    # discretizing trained logits net and evaluating
    discrete_net = logit_net.generate_discrete_networks("sample")
    discrete_net = discrete_net.to(device)
    discrete_net.evaluate_and_save_to_disk(test_loader, "FashionMnistConv-ex4.1_trained_discretized_ternary_sample")
    discrete_net = logit_net.generate_discrete_networks("argmax")
    discrete_net = discrete_net.to(device)
    discrete_net.evaluate_and_save_to_disk(test_loader, "FashionMnistConv-ex4.1_trained_discretized_ternary_argmax")


if __name__ == "__main__":
    train_model("FashionMnistConv-real-conv-trained-2020-3-4--h11m53")

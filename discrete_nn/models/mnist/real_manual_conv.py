"""
This module implements the real valued (non convolutionary) network for the mnist dataset
"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import os
import pickle
from sklearn.metrics import accuracy_score

from discrete_nn.models.base_model import AlternateDiscretizationBaseModel
from discrete_nn.dataset.mnist import MNIST
from discrete_nn.settings import model_path
from discrete_nn.layers.Flatten import Flatten

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device == "cuda:0":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class MnistAlternateDiscreteConv(AlternateDiscretizationBaseModel):
    """
    Real valued (non convolutionary) network for the mnist dataset
    """

    def __init__(self):
        super().__init__()
        # defining all the network's layers
        self.netlayers = torch.nn.Sequential(

            torch.nn.Conv2d(1, 32, 5, stride=1, bias=False, padding=2),
            torch.nn.BatchNorm2d(32, momentum=0.1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(2),
            #
            torch.nn.Dropout(p=0.2),
            torch.nn.Conv2d(32, 64, 5, stride=1, bias=False, padding=2),
            torch.nn.BatchNorm2d(64, momentum=0.1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(2),

            #
            Flatten(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(3136, 512),
            torch.nn.BatchNorm1d(512, momentum=0.1),
            torch.nn.Tanh(),
            #
            torch.nn.Linear(512, 10)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        self.loss_funct = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # takes image vector
        return self.netlayers(x)

    def set_net_parameters(self, param_dict):
        self.state_dict()["netlayers.0.weight"] = param_dict["L1_Conv_W"]
        self.state_dict()["netlayers.1.weight"] = param_dict["L1_BatchNorm_W"]
        self.state_dict()["netlayers.1.bias"] = param_dict["L1_BatchNorm_b"]
        self.state_dict()["netlayers.5.weight"] = param_dict["L2_Conv_W"]
        self.state_dict()["netlayers.6.weight"] = param_dict["L2_BatchNorm_W"]
        self.state_dict()["netlayers.6.bias"] = param_dict["L2_BatchNorm_b"]
        self.state_dict()["netlayers.11.weight"] = param_dict["L3_Linear_W"]
        self.state_dict()["netlayers.11.bias"] = param_dict["L3_Linear_b"].reshape(-1)
        self.state_dict()["netlayers.12.weight"] = param_dict["L3_BatchNorm_W"]
        self.state_dict()["netlayers.12.bias"] = param_dict["L3_BatchNorm_b"]
        self.state_dict()["netlayers.14.weight"] = param_dict["L4_Linear_W"]
        self.state_dict()["netlayers.14.bias"] = param_dict["L4_Linear_b"].reshape(-1)

    def get_net_parameters(self):
        """:returns a dictionary with the trainable parameters"""
        internal_dict = {name: value for name, value in self.named_parameters()}

        repr_dict = dict()
        repr_dict["L1_Conv_W"] = internal_dict["netlayers.0.weight"]
        repr_dict["L1_BatchNorm_W"] = internal_dict["netlayers.1.weight"]
        repr_dict["L1_BatchNorm_b"] = internal_dict["netlayers.1.bias"]
        repr_dict["L2_Conv_W"] = internal_dict["netlayers.5.weight"]
        repr_dict["L2_BatchNorm_W"] = internal_dict["netlayers.6.weight"]
        repr_dict["L2_BatchNorm_b"] = internal_dict["netlayers.6.bias"]
        repr_dict["L3_Linear_W"] = internal_dict["netlayers.11.weight"]
        repr_dict["L3_Linear_b"] = internal_dict["netlayers.11.bias"].reshape(-1, 1)
        repr_dict["L3_BatchNorm_W"] = internal_dict["netlayers.12.weight"]
        repr_dict["L3_BatchNorm_b"] = internal_dict["netlayers.12.bias"]
        repr_dict["L4_Linear_W"] = internal_dict["netlayers.14.weight"]
        repr_dict["L4_Linear_b"] = internal_dict["netlayers.14.bias"].reshape(-1, 1)
        return repr_dict

    def discretize(self):
        # Manually adjust the weights
        modules = self._modules["netlayers"]._modules
        layerIndices = ["0", "5", "11", "14"]
        for mod in layerIndices:
            layer = modules[str(mod)]
            if hasattr(layer, "weight"):
                data = layer.weight
                for x in range(0, len(data)):
                    min = torch.min(data[x])
                    max = torch.max(data[x])
                    data[x] = 2 * ((data[x] - min) / (max - min)) - 1
                    data[x] = torch.round(data[x])     #Use this for Ternary
                    #data[x] = torch.round(data[x] / 0.5) * 0.5  # Use this for Quinary
        return


class DatasetMNIST(Dataset):
    """
    Dataset for pytorch's DataLoader
    """

    def __init__(self, x, y):
        self.x = torch.from_numpy(x) * 2 - 1
        self.y = torch.from_numpy(y).long()
        self.x = self.x.reshape((self.x.shape[0], 1, 28, 28))
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item_inx):
        return self.x[item_inx], self.y[item_inx]


def train_model():
    # basic dataset holder
    mnist = MNIST()
    # creates the dataloader for pytorch
    batch_size = 100
    train_loader = DataLoader(dataset=DatasetMNIST(mnist.x_train, mnist.y_train), batch_size=batch_size,
                              shuffle=True)
    validation_loader = DataLoader(dataset=DatasetMNIST(mnist.x_val, mnist.y_val), batch_size=batch_size,
                                   shuffle=False)
    test_loader = DataLoader(dataset=DatasetMNIST(mnist.x_test, mnist.y_test), batch_size=batch_size,
                             shuffle=False)

    net = MnistAlternateDiscreteConv()
    net = net.to(device)

    num_epochs = 100
    # will save metrics and model to disk. returns the path to metrics and saved model
    return net.train_model(train_loader, validation_loader, test_loader, num_epochs, model_name="MNIST-alt-conv")


if __name__ == "__main__":
    print('Using device:', device)
    train_model()
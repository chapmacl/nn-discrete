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
from sklearn.metrics import accuracy_score, classification_report

from discrete_nn.dataset.mnist import MNIST
from discrete_nn.settings import model_path
from discrete_nn.layers.types import ValueTypes
from discrete_nn.layers.logit_linear import TernaryLinear
from discrete_nn.layers.local_reparametrization import LocalReparametrization
from discrete_nn.models.mnist_pi.real import MnistPiReal
from discrete_nn.models.base_model import BaseModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device == "cuda:0":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_default_dtype(torch.Float16)



class MnistPiTernaryTanh(BaseModel):
    """
    Real valued (non convolutionary) network for the mnist dataset
    """

    def __init__(self, real_model_params):
        """

        :param real_model_params: a dictionary containing the real weights of the pretrained model
        """
        super().__init__()

        s1_l1_dropout = torch.nn.Dropout(p=0.1)
        s2_l1_linear = TernaryLinear(784, ValueTypes.REAL, 1200, real_model_params["L1_Linear_W"],
                                     real_model_params["L1_Linear_b"])
        s3_l1_repar = LocalReparametrization(1200, ValueTypes.GAUSSIAN) # outputs a value and not a dist.
        s4_l1_batchnorm = torch.nn.BatchNorm1d(1200, momentum=0.1)
        s5_l1_tanh = torch.nn.Tanh()

        s6_l2_dropout = torch.nn.Dropout(p=0.2)
        s7_l2_linear = TernaryLinear(1200, ValueTypes.REAL, 1200, real_model_params["L2_Linear_W"],
                                     real_model_params["L2_Linear_b"])
        s8_l2_repar = LocalReparametrization(1200, ValueTypes.GAUSSIAN)  # outputs a value and not a dist.
        s9_l2_batchnorm = torch.nn.BatchNorm1d(1200, momentum=0.1)
        s10_l2_tanh = torch.nn.Tanh()

        s6_l3_dropout = torch.nn.Dropout(p=0.3)
        s7_l3_linear = TernaryLinear(1200, ValueTypes.REAL, 10, real_model_params["L3_Linear_W"],
                                     real_model_params["L3_Linear_b"], normalize_activations=True)
        s8_l3_repar = LocalReparametrization(10, ValueTypes.GAUSSIAN)
        # defining all the network's layers
        self.netlayers = torch.nn.Sequential(
            s1_l1_dropout,
            s2_l1_linear,
            s3_l1_repar,
            s4_l1_batchnorm,
            s5_l1_tanh,
            s6_l2_dropout,
            s7_l2_linear,
            s8_l2_repar,
            s9_l2_batchnorm,
            s10_l2_tanh,
            s6_l3_dropout,
            s7_l3_linear,
            s8_l3_repar)  # last layer outputs the unnormalized loglikelihood used by the softmax later

        """setting parameters of torch layers here"""
        self.state_dict()['netlayers.3.weight'][:] = real_model_params["L1_BatchNorm_W"]
        self.state_dict()['netlayers.3.bias'][:] = real_model_params["L1_BatchNorm_b"]
        self.state_dict()['netlayers.8.weight'][:] = real_model_params["L2_BatchNorm_W"]
        self.state_dict()['netlayers.8.bias'][:] = real_model_params["L2_BatchNorm_b"]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-8)
        print(self.optimizer)
        self.loss_funct = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # takes image vector
        return self.netlayers(x)

    def get_net_parameters(self):
        return self.state_dict()

    def set_net_parameters(self, param_dict):
        for k, v in param_dict.items():
            self.state_dict()[k][:] = v

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
            "L1_BatchNorm_W": self.state_dict()['netlayers.3.weight'],
            "L1_BatchNorm_b": self.state_dict()['netlayers.3.bias'],
            "L2_Linear_W": l2_sampled_w,
            "L2_Linear_b": l2_sampled_b,
            "L2_BatchNorm_W": self.state_dict()['netlayers.8.weight'],
            "L2_BatchNorm_b": self.state_dict()['netlayers.8.bias'],
            "L3_Linear_W": l3_sampled_w,
            "L3_Linear_b": l3_sampled_b
        }

        real_net = MnistPiReal()
        real_net.to(device)
        real_net.set_net_parameters(state_dict)
        return real_net

class DatasetMNIST(Dataset):
    """
    Dataset for pytorch's DataLoader
    """

    def __init__(self, x, y):
        self.x = torch.from_numpy(x) * 2 - 1
        self.y = torch.from_numpy(y).long()
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item_inx):
        return self.x[item_inx], self.y[item_inx]


def train_model():
    batch_size = 100
    # basic dataset holder
    mnist = MNIST()
    # creates the dataloader for pytorch
    train_loader = DataLoader(dataset=DatasetMNIST(mnist.x_train, mnist.y_train), batch_size=batch_size,
                              shuffle=True)
    validation_loader = DataLoader(dataset=DatasetMNIST(mnist.x_val, mnist.y_val), batch_size=batch_size,
                                   shuffle=False)
    test_loader = DataLoader(dataset=DatasetMNIST(mnist.x_test, mnist.y_test), batch_size=batch_size,
                             shuffle=False)

    print('Using device:', device)

    real_model_param_path = os.path.join(model_path, "MnistPiReal-real-trained-2020-2-2--h17m58",
                                         "MnistPiReal.param.pickle")
    with open(real_model_param_path, "rb") as f:
        real_param = pickle.load(f)
        logit_net = MnistPiTernaryTanh(real_param)
    logit_net = logit_net.to(device)
    # discretizing and evaluating
    # todo should probably generate several sampled ones?
    discrete_net = logit_net.generate_discrete_networks("sample")
    discrete_net = discrete_net.to(device)
    discrete_net.evaluate_and_save_to_disk(test_loader, "ex3.1_untrained_discretized_ternary_sample")
    discrete_net = logit_net.generate_discrete_networks("argmax")
    discrete_net = discrete_net.to(device)
    discrete_net.evaluate_and_save_to_disk(test_loader, "ex3.1_untrained_discretized_ternary_argmax")
    del discrete_net
    # evaluate first logit model before training, train and evaluate again
    
    logit_net.train_model(train_loader, validation_loader, test_loader, 100, "logits_ternary_tanh", True)

    # discretizing trained logits net and evaluating
    discrete_net = logit_net.generate_discrete_networks("sample")
    discrete_net = discrete_net.to(device)
    discrete_net.evaluate_and_save_to_disk(test_loader, "ex4.1_trained_discretized_ternary_sample")
    discrete_net = logit_net.generate_discrete_networks("argmax")
    discrete_net = discrete_net.to(device)
    discrete_net.evaluate_and_save_to_disk(test_loader, "ex4.1_trained_discretized_ternary_argmax")

if __name__ == "__main__":
    train_model()
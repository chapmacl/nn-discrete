"""
This module implements the real valued (non convolutionary) network for the conv dataset
"""
import torch

from discrete_nn.layers.type_defs import ValueTypes, DiscreteWeights
from discrete_nn.layers.logit_linear import LogitLinear
from discrete_nn.layers.conv import LogitConv
from discrete_nn.layers.local_reparametrization import LocalReparametrization
from discrete_nn.models.conv.real import ConvReal
from discrete_nn.models.base_model import LogitModel

from discrete_nn.layers.Flatten import Flatten


class ConvLogitTanh(LogitModel):
    """
    Real valued (non convolutionary) network for the conv dataset
    """

    def __init__(self, real_model_params, discrete_weights: DiscreteWeights):
        """

        :param real_model_params: a dictionary containing the real weights of the pretrained model
        """
        super().__init__()
        self.netlayers = torch.nn.Sequential(
            LogitConv(1, ValueTypes.REAL, 32, 5, 1, real_model_params["L1_Conv_W"], None, discrete_weights),
            # torch.nn.Conv2d(1, 32, 5, stride=1, bias=False),
            LocalReparametrization(),
            torch.nn.BatchNorm2d(32, track_running_stats=False),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(2),
            #
            torch.nn.Dropout(p=0.2),
            # torch.nn.Conv2d(32, 64, 5, stride=1, bias=False),
            LogitConv(32, ValueTypes.REAL, 64, 5, 1, real_model_params["L2_Conv_W"], None, discrete_weights),
            LocalReparametrization(),
            torch.nn.BatchNorm2d(64, track_running_stats=False),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(2),

            #
            Flatten(),
            torch.nn.Dropout(p=0.3),
            LogitLinear(3136, ValueTypes.REAL, 512, real_model_params["L3_Linear_W"],
                        None, discrete_weights, normalize_activations=False),
            LocalReparametrization(),
            torch.nn.BatchNorm1d(512, track_running_stats=False),
            torch.nn.Tanh(),
            #
            LogitLinear(512, ValueTypes.REAL, 10, real_model_params["L4_Linear_W"], real_model_params["L4_Linear_b"],
                        discrete_weights, normalize_activations=True),
            LocalReparametrization(),
        )

        self.state_dict()["netlayers.2.weight"] = real_model_params["L1_BatchNorm_W"]
        self.state_dict()["netlayers.2.bias"] = real_model_params["L1_BatchNorm_b"]
        self.state_dict()["netlayers.8.weight"] = real_model_params["L2_BatchNorm_W"]
        self.state_dict()["netlayers.8.bias"] = real_model_params["L2_BatchNorm_b"]
        self.state_dict()["netlayers.15.weight"] = real_model_params["L3_BatchNorm_W"]
        self.state_dict()["netlayers.15.bias"] = real_model_params["L3_BatchNorm_b"]

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

    def generate_discrete_networks(self, method: str) -> ConvReal:
        """

        :param method: sample or argmax
        :return:
        """
        # state dicts
        l1_layer: LogitConv = self.netlayers[0]
        l1_sampled_w, l1_sampled_b = l1_layer.generate_discrete_network(method)
        l2_layer: LogitConv = self.netlayers[6]
        l2_sampled_w, l2_sampled_b = l2_layer.generate_discrete_network(method)
        l3_layer: LogitLinear = self.netlayers[13]
        l3_sampled_w, l3_sampled_b = l3_layer.generate_discrete_network(method)
        l4_layer: LogitLinear = self.netlayers[17]
        l4_sampled_w, l4_sampled_b = l4_layer.generate_discrete_network(method)
        state_dict = {
            "L1_Conv_W": l1_sampled_w,
            "L1_BatchNorm_W": self.state_dict()['netlayers.2.weight'],
            "L1_BatchNorm_b": self.state_dict()['netlayers.2.bias'],
            "L2_Conv_W": l2_sampled_w,
            "L2_BatchNorm_W": self.state_dict()['netlayers.8.weight'],
            "L2_BatchNorm_b": self.state_dict()['netlayers.8.bias'],
            "L3_Linear_W": l3_sampled_w,
            "L3_BatchNorm_W": self.state_dict()["netlayers.15.weight"],
            "L3_BatchNorm_b": self.state_dict()["netlayers.15.bias"],
            "L4_Linear_W": l4_sampled_w,
            "L4_Linear_b": l4_sampled_b
        }
        real_net = ConvReal()
        real_net.set_net_parameters(state_dict)
        return real_net


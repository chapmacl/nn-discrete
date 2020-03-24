"""
This module implements the real valued (non convolutionary) network for the conv dataset
"""
import torch

from discrete_nn.layers.type_defs import ValueTypes, DiscreteWeights, InputFormat
from discrete_nn.layers.logit_linear import LogitLinear
from discrete_nn.layers.conv import LogitConv
from discrete_nn.layers.sign import DistributionSign
from discrete_nn.layers.pool import DistributionMaxPool
from discrete_nn.layers.distribution_batchnorm import DistributionBatchnorm
from discrete_nn.layers.local_reparametrization import LocalReparametrization
from discrete_nn.models.conv.discrete_sign import ConvDiscreteSign
from discrete_nn.models.base_model import LogitModel
from discrete_nn.layers.Flatten import Flatten


class ConvLogitSign(LogitModel):
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
            DistributionMaxPool(2),  # torch.nn.MaxPool2d(2),
            DistributionBatchnorm(InputFormat.FEATURE_MAP, 32, real_model_params["L1_BatchNorm_W"],
                                  real_model_params["L1_BatchNorm_b"]),  # torch.nn.BatchNorm2d(32, momentum=0.1),
            DistributionSign(InputFormat.FEATURE_MAP),
            #
            LocalReparametrization(),
            torch.nn.Dropout(p=0.2),

            # torch.nn.Conv2d(32, 64, 5, stride=1, bias=False),
            LogitConv(32, ValueTypes.REAL, 64, 5, 1, real_model_params["L2_Conv_W"], None, discrete_weights),
            DistributionMaxPool(2),  # torch.nn.MaxPool2d(2),
            DistributionBatchnorm(InputFormat.FEATURE_MAP, 64, real_model_params["L2_BatchNorm_W"],
                                  real_model_params["L2_BatchNorm_b"]),
            DistributionSign(InputFormat.FEATURE_MAP),
            LocalReparametrization(),
            #
            Flatten(),
            torch.nn.Dropout(p=0.3),
            LogitLinear(3136, ValueTypes.REAL, 512, real_model_params["L3_Linear_W"],
                        None, discrete_weights, normalize_activations=False),
            DistributionBatchnorm(InputFormat.FLAT_ARRAY, 512, real_model_params["L3_BatchNorm_W"],
                                  real_model_params["L3_BatchNorm_b"]),  # torch.nn.BatchNorm1d(512, momentum=0.1),
            DistributionSign(InputFormat.FLAT_ARRAY),
            LocalReparametrization(),
            #
            LogitLinear(512, ValueTypes.REAL, 10, real_model_params["L4_Linear_W"], real_model_params["L4_Linear_b"],
                        discrete_weights, normalize_activations=True),
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
        self.load_state_dict(param_dict, strict=False)

    def generate_discrete_networks(self, method: str) -> ConvDiscreteSign:
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
            "L1_BatchNorm_W": self.state_dict()['netlayers.2.gamma'],
            "L1_BatchNorm_b": self.state_dict()['netlayers.2.beta'],
            "L2_Conv_W": l2_sampled_w,
            "L2_BatchNorm_W": self.state_dict()['netlayers.8.gamma'],
            "L2_BatchNorm_b": self.state_dict()['netlayers.8.beta'],
            "L3_Linear_W": l3_sampled_w,
            "L3_BatchNorm_W": self.state_dict()["netlayers.14.gamma"],
            "L3_BatchNorm_b": self.state_dict()["netlayers.14.beta"],
            "L4_Linear_W": l4_sampled_w,
            "L4_Linear_b": l4_sampled_b
        }
        real_net = ConvDiscreteSign()
        real_net.set_net_parameters(state_dict)
        return real_net

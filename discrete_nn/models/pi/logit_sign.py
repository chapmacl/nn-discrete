"""
This module implements the real valued (non convolutionary) network for the PI model
"""
import torch

from discrete_nn.layers.type_defs import ValueTypes, InputFormat, DiscreteWeights
from discrete_nn.layers.logit_linear import LogitLinear
from discrete_nn.layers.sign import DistributionSign
from discrete_nn.models.pi.discrete_sign import PiDiscreteSign
from discrete_nn.layers.distribution_batchnorm import DistributionBatchnorm
from discrete_nn.layers.local_reparametrization import LocalReparametrization
from discrete_nn.models.base_model import LogitModel


class PiLogitSign(LogitModel):
    """
    Real valued (non convolutionary) network for the Pi architecture.
    """

    def __init__(self, real_model_params, discrete_weights: DiscreteWeights):
        """

        :param real_model_params: a dictionary containing the real weights of the pretrained model
        """
        super().__init__()
        # defining all the network's layers
        self.netlayers = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            LogitLinear(784, ValueTypes.REAL, 1200, real_model_params["L1_Linear_W"],
                        None, discrete_weights),
            DistributionBatchnorm(InputFormat.FLAT_ARRAY, 1200, real_model_params["L1_BatchNorm_W"],
                                  real_model_params["L1_BatchNorm_b"]),  # ?, momentum=0.1)
            DistributionSign(InputFormat.FLAT_ARRAY),
            LocalReparametrization(),  # outputs a value and not a dist.

            torch.nn.Dropout(p=0.2),
            LogitLinear(1200, ValueTypes.REAL, 1200, real_model_params["L2_Linear_W"],
                        None, discrete_weights),
            DistributionBatchnorm(InputFormat.FLAT_ARRAY, 1200, real_model_params["L2_BatchNorm_W"],
                                  real_model_params["L2_BatchNorm_b"]),  # ?, momentum=0.1)
            DistributionSign(InputFormat.FLAT_ARRAY),
            LocalReparametrization(),  # outputs a value and not a dist.

            torch.nn.Dropout(p=0.3),
            LogitLinear(1200, ValueTypes.REAL, 10, real_model_params["L3_Linear_W"],
                        real_model_params["L3_Linear_b"], discrete_weights, normalize_activations=True),
            LocalReparametrization())  # last layer outputs the unnormalized loglikelihood used by the softmax later

        # optimizer configurations
        opt_param = [
            {"params": self.netlayers[2].parameters(), 'lr':1e-3},
            {"params": self.netlayers[7].parameters(), 'lr':1e-3},
            {"params": self.netlayers[1].parameters()},  # logit layers
            {"params": self.netlayers[6].parameters()},
            {"params": self.netlayers[11].parameters()},
        ]
        self.optimizer = torch.optim.Adam(opt_param, lr=1e-2, weight_decay=1e-10)
        print(self.optimizer)
        self.loss_funct = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # takes image vector
        return self.netlayers(x)

    def generate_discrete_networks(self, method: str) -> PiDiscreteSign:
        """

        :param method: sample or argmax
        :return:
        """
        # state dicts
        l1_layer: LogitLinear = self.netlayers[1]
        l1_sampled_w, l1_sampled_b = l1_layer.generate_discrete_network(method)
        l2_layer: LogitLinear = self.netlayers[6]
        l2_sampled_w, l2_sampled_b = l2_layer.generate_discrete_network(method)
        l3_layer: LogitLinear = self.netlayers[11]
        l3_sampled_w, l3_sampled_b = l3_layer.generate_discrete_network(method)
        state_dict = {
            "L1_Linear_W": l1_sampled_w,
            "L1_BatchNorm_W": self.state_dict()['netlayers.2.gamma'],
            "L1_BatchNorm_b": self.state_dict()['netlayers.2.beta'],
            "L2_Linear_W": l2_sampled_w,
            "L2_BatchNorm_W": self.state_dict()['netlayers.7.gamma'],
            "L2_BatchNorm_b": self.state_dict()['netlayers.7.beta'],
            "L3_Linear_W": l3_sampled_w,
            "L3_Linear_b": l3_sampled_b
        }

        discrete_net = PiDiscreteSign()
        discrete_net.set_net_parameters(state_dict)
        return discrete_net

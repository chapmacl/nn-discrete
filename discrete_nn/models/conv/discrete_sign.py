"""
implements a discrete model sign activated architecture for conv convolutional
"""
import torch

from discrete_nn.models.base_model import BaseModel
from discrete_nn.layers.sign import DiscreteSign
from discrete_nn.layers.linear import Linear
from discrete_nn.layers.Flatten import Flatten


class ConvDiscreteSign(BaseModel):
    """
    Real valued (non convolutionary) network for the conv dataset
    """

    def __init__(self):
        super().__init__()
        # defining all the network's layers
        self.netlayers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 5, stride=1, bias=False, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(32, track_running_stats=False),
            DiscreteSign(),
            #
            torch.nn.Dropout(p=0.2),

            torch.nn.Conv2d(32, 64, 5, stride=1, bias=False, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(64, track_running_stats=False),
            DiscreteSign(),
            #
            Flatten(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(3136, 512, bias=False),
            torch.nn.BatchNorm1d(512, track_running_stats=False),
            DiscreteSign(),
            #
            Linear(512, 10, normalize_activations=True))

        self.loss_funct = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # takes image vector
        return self.netlayers(x)

    def set_net_parameters(self, param_dict):
        new_state_dict = {
            "netlayers.0.weight": param_dict["L1_Conv_W"].clone().detach(),
            "netlayers.2.weight": param_dict["L1_BatchNorm_W"].clone().detach(),
            "netlayers.2.bias": param_dict["L1_BatchNorm_b"].clone().detach(),
            "netlayers.5.weight": param_dict["L2_Conv_W"].clone().detach(),
            "netlayers.7.weight": param_dict["L2_BatchNorm_W"].clone().detach(),
            "netlayers.7.bias": param_dict["L2_BatchNorm_b"].clone().detach(),
            "netlayers.11.weight": param_dict["L3_Linear_W"].clone().detach(),
            "netlayers.12.weight": param_dict["L3_BatchNorm_W"].clone().detach(),
            "netlayers.12.bias": param_dict["L3_BatchNorm_b"].clone().detach(),
            "netlayers.14.weight": param_dict["L4_Linear_W"].clone().detach(),
            "netlayers.14.bias": param_dict["L4_Linear_b"].clone().detach().reshape(-1)
        }
        # we do not want to load batch norm parameters so strict = False
        self.load_state_dict(new_state_dict, strict=False)

    def get_net_parameters(self):
        """:returns a dictionary with the trainable parameters"""
        internal_dict = {name: value for name, value in self.named_parameters()}

        repr_dict = dict()
        repr_dict["L1_Conv_W"] = internal_dict["netlayers.0.weight"]
        repr_dict["L1_BatchNorm_W"] = internal_dict["netlayers.2.weight"]
        repr_dict["L1_BatchNorm_b"] = internal_dict["netlayers.2.bias"]
        repr_dict["L2_Conv_W"] = internal_dict["netlayers.5.weight"]
        repr_dict["L2_BatchNorm_W"] = internal_dict["netlayers.7.weight"]
        repr_dict["L2_BatchNorm_b"] = internal_dict["netlayers.7.bias"]
        repr_dict["L3_Linear_W"] = internal_dict["netlayers.11.weight"]
        repr_dict["L3_BatchNorm_W"] = internal_dict["netlayers.12.weight"]
        repr_dict["L3_BatchNorm_b"] = internal_dict["netlayers.12.bias"]
        repr_dict["L4_Linear_W"] = internal_dict["netlayers.14.weight"]
        repr_dict["L4_Linear_b"] = internal_dict["netlayers.14.bias"].reshape(-1, 1)
        return repr_dict

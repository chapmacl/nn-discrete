"""
This module implements the real valued (non convolutionary) network for the conv dataset
"""
import torch

from discrete_nn.models.base_model import BaseModel
from discrete_nn.layers.Flatten import Flatten


class ConvReal(BaseModel):
    """
    Real valued convolutional network
    """

    def __init__(self):
        super().__init__()
        # defining all the network's layers
        self.netlayers = torch.nn.Sequential(

            torch.nn.Conv2d(1, 32, 5, stride=1, bias=False, padding=2),
            torch.nn.BatchNorm2d(32, track_running_stats=False),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(2),
            #
            torch.nn.Dropout(p=0.2),
            torch.nn.Conv2d(32, 64, 5, stride=1, bias=False, padding=2),
            torch.nn.BatchNorm2d(64, track_running_stats=False),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(2),
            #
            Flatten(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(3136, 512, bias=False),
            torch.nn.BatchNorm1d(512, track_running_stats=False),
            torch.nn.Tanh(),
            #
            torch.nn.Linear(512, 10)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-6)
        self.loss_funct = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # takes image vector
        return self.netlayers(x)

    def set_net_parameters(self, param_dict):
        new_state_dict = {
            "netlayers.0.weight": param_dict["L1_Conv_W"],
            "netlayers.1.weight": param_dict["L1_BatchNorm_W"],
            "netlayers.1.bias": param_dict["L1_BatchNorm_b"],
            "netlayers.5.weight": param_dict["L2_Conv_W"],
            "netlayers.6.weight": param_dict["L2_BatchNorm_W"],
            "netlayers.6.bias": param_dict["L2_BatchNorm_b"],
            "netlayers.11.weight": param_dict["L3_Linear_W"],
            "netlayers.12.weight": param_dict["L3_BatchNorm_W"],
            "netlayers.12.bias": param_dict["L3_BatchNorm_b"],
            "netlayers.14.weight": param_dict["L4_Linear_W"],
            "netlayers.14.bias": param_dict["L4_Linear_b"].reshape(-1)
        }
        self.load_state_dict(new_state_dict, strict=False)

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
        repr_dict["L3_BatchNorm_W"] = internal_dict["netlayers.12.weight"]
        repr_dict["L3_BatchNorm_b"] = internal_dict["netlayers.12.bias"]
        repr_dict["L4_Linear_W"] = internal_dict["netlayers.14.weight"]
        repr_dict["L4_Linear_b"] = internal_dict["netlayers.14.bias"].reshape(-1, 1)
        return repr_dict


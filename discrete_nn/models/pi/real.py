"""
This module implements the real valued (non convolutionary) network for the PI model
"""
import torch

from discrete_nn.models.base_model import BaseModel


class PiReal(BaseModel):
    """
    Real valued (non convolutionary) network for the PI model
    """

    def __init__(self):
        """
        Initializes a Pi Architecture network
        """
        super().__init__()
        # defining all the network's layers
        self.netlayers = torch.nn.Sequential(

            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(784, 1200, bias=False),
            torch.nn.BatchNorm1d(1200, track_running_stats=False),
            # momentum equivalent to alpha on reference impl.
            # should batch normalization be here or after the activation function ?
            torch.nn.Tanh(),
            #
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(1200, 1200, bias=False),
            torch.nn.BatchNorm1d(1200, track_running_stats=False),
            torch.nn.Tanh(),
            #
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(1200, 10))  # last layer outputs the unnormalized loglikelihood used by the softmax later
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        self.loss_funct = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.netlayers(x)

    def set_net_parameters(self, param_dict):
        new_state_dict = {
            "netlayers.1.weight": param_dict["L1_Linear_W"],
            'netlayers.2.weight': param_dict["L1_BatchNorm_W"],
            'netlayers.2.bias': param_dict["L1_BatchNorm_b"],
            "netlayers.5.weight": param_dict["L2_Linear_W"],
            'netlayers.6.weight': param_dict["L2_BatchNorm_W"],
            'netlayers.6.bias': param_dict["L2_BatchNorm_b"],
            "netlayers.9.weight": param_dict["L3_Linear_W"],
            "netlayers.9.bias": param_dict["L3_Linear_b"].reshape(-1)
        }
        self.load_state_dict(new_state_dict, strict=False)

    def get_net_parameters(self):
        """:returns a dictionary with the trainable parameters"""
        internal_dict = {name: value for name, value in self.named_parameters()}
        repr_dict = dict()
        repr_dict["L1_Linear_W"] = internal_dict["netlayers.1.weight"]
        repr_dict["L1_BatchNorm_W"] = internal_dict["netlayers.2.weight"]
        repr_dict["L1_BatchNorm_b"] = internal_dict["netlayers.2.bias"]
        repr_dict["L2_Linear_W"] = internal_dict["netlayers.5.weight"]
        repr_dict["L2_BatchNorm_W"] = internal_dict["netlayers.6.weight"]
        repr_dict["L2_BatchNorm_b"] = internal_dict["netlayers.6.bias"]
        repr_dict["L3_Linear_W"] = internal_dict["netlayers.9.weight"]
        repr_dict["L3_Linear_b"] = internal_dict["netlayers.9.bias"].reshape(-1, 1)
        return repr_dict

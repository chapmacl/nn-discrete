"""
implements a discrete model sign activated PI architecture
"""
import torch
from discrete_nn.layers.type_defs import ValueTypes, InputFormat
from discrete_nn.models.base_model import BaseModel
from discrete_nn.layers.sign import DiscreteSign
from discrete_nn.layers.linear import Linear
from discrete_nn.layers.distribution_batchnorm import DistributionBatchnorm


class PiDiscreteSign(BaseModel):
    """
    Discrete valued PI architecture network
    """

    def __init__(self):
        """
        """
        super().__init__()
        # defining all the network's layers
        self.netlayers = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(784, 1200, bias=False),
            DistributionBatchnorm(InputFormat.FLAT_ARRAY, ValueTypes.REAL, 1200, None, None),
            #torch.nn.BatchNorm1d(1200, track_running_stats=False),
            # momentum equivalent to alpha on reference impl.
            # should batch normalization be here or after the activation function ?
            DiscreteSign(),
            #
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(1200, 1200, bias=False),
            DistributionBatchnorm(InputFormat.FLAT_ARRAY, ValueTypes.REAL, 1200, None, None),
            #torch.nn.BatchNorm1d(1200, track_running_stats=False),
            DiscreteSign(),
            #
            torch.nn.Dropout(p=0.4),
            Linear(1200, 10, normalize_activations=True))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        self.loss_funct = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # takes image vector
        x = x
        return self.netlayers(x)

    def set_net_parameters(self, param_dict):
        new_stat_dict = {
          "netlayers.1.weight": param_dict["L1_Linear_W"],
          'netlayers.2.gamma': param_dict["L1_BatchNorm_W"],
          'netlayers.2.beta': param_dict["L1_BatchNorm_b"],
          "netlayers.5.weight": param_dict["L2_Linear_W"],
          'netlayers.6.gamma': param_dict["L2_BatchNorm_W"],
          'netlayers.6.beta': param_dict["L2_BatchNorm_b"],
          "netlayers.9.weight": param_dict["L3_Linear_W"],
          "netlayers.9.bias": param_dict["L3_Linear_b"].reshape(-1)
        }
        self.load_state_dict(new_stat_dict, strict=False)
        return

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



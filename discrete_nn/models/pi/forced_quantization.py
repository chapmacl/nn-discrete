"""
This module implements the real valued (non convolutionary) network for the mnist dataset
"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


from discrete_nn.models.base_model import ForcedQuantizationBaseModel
from discrete_nn.dataset.mnist import MNIST


class PiForcedQuantization(ForcedQuantizationBaseModel):
    """
    Real valued (non convolutionary) network for Pi arquitecture with forced quantization
    """

    def __init__(self):
        super().__init__()
        # defining all the network's layers
        self.netlayers = torch.nn.Sequential(

            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(784, 1200, bias=False),
            torch.nn.BatchNorm1d(1200, momentum=0.1),
            # momentum equivalent to alpha on reference impl.
            # should batch normalization be here or after the activation function ?
            torch.nn.Tanh(),
            #
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(1200, 1200, bias=False),
            torch.nn.BatchNorm1d(1200, momentum=0.1),
            torch.nn.Tanh(),
            #
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(1200, 10))  # last layer outputs the unnormalized loglikelihood used by the softmax later

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        self.loss_funct = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # takes image vector
        return self.netlayers(x)

    def get_net_parameters(self):
        """:returns a dictionary with the trainable parameters"""
        internal_dict = {name: value for name, value in self.named_parameters()}

        repr_dict = dict()
        repr_dict["L1_Linear_W"] = internal_dict["netlayers.1.weight"]
        #repr_dict["L1_Linear_b"] = internal_dict["netlayers.1.bias"].reshape(-1, 1)
        repr_dict["L1_BatchNorm_W"] = internal_dict["netlayers.2.weight"]
        repr_dict["L1_BatchNorm_b"] = internal_dict["netlayers.2.bias"]
        repr_dict["L2_Linear_W"] = internal_dict["netlayers.5.weight"]
        #repr_dict["L2_Linear_b"] = internal_dict["netlayers.5.bias"].reshape(-1, 1)
        repr_dict["L2_BatchNorm_W"] = internal_dict["netlayers.6.weight"]
        repr_dict["L2_BatchNorm_b"] = internal_dict["netlayers.6.bias"]
        repr_dict["L3_Linear_W"] = internal_dict["netlayers.9.weight"]
        repr_dict["L3_Linear_b"] = internal_dict["netlayers.9.bias"].reshape(-1, 1)
        return repr_dict

    def discretize(self):
        # Manually adjust the weights
        modules = self._modules["netlayers"]._modules
        layerIndices = ["1", "5", "9"]
        for mod in layerIndices:
            layer = modules[str(mod)]
            if hasattr(layer, "weight"):
                data = layer.weight
                for x in range(0, len(data)):
                    min = torch.min(data[x])
                    max = torch.max(data[x])
                    data[x] = 2 * ((data[x] - min) / (max - min)) - 1
                    # data[x] = torch.round(data[x])     #Use this for Ternary
                    data[x] = torch.round(data[x] / 0.5) * 0.5  # Use this for Quinary
        return



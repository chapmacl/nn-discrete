"""
This module implements the real valued (non convolutionary) network for the mnist dataset
"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from discrete_nn.dataset.mnist import MNIST
from discrete_nn.models.base_model import BaseModel

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
if device == "cuda:0":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class MnistPiReal(BaseModel):
    """
    Real valued (non convolutionary) network for the mnist dataset
    """

    def __init__(self, weights = None):
        """

        :param weights: if not none contains the weighs for the networks layers
        """
        super().__init__()
        # defining all the network's layers
        self.netlayers = torch.nn.Sequential(

            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(784, 1200),
            torch.nn.BatchNorm1d(1200, momentum=0.1),
            # momentum equivalent to alpha on reference impl.
            # should batch normalization be here or after the activation function ?
            torch.nn.Tanh(),
            #
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(1200, 1200),
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

    def set_net_parameters(self, param_dict):
        self.state_dict()["netlayers.1.weight"][:] = param_dict["L1_Linear_W"]
        self.state_dict()["netlayers.1.bias"][:] = param_dict["L1_Linear_b"].reshape(-1)
        self.state_dict()['netlayers.2.weight'][:] = param_dict["L1_BatchNorm_W"]
        self.state_dict()['netlayers.2.bias'][:] = param_dict["L1_BatchNorm_b"]
        self.state_dict()["netlayers.5.weight"][:] = param_dict["L2_Linear_W"]
        self.state_dict()["netlayers.5.bias"][:] = param_dict["L2_Linear_b"].reshape(-1)
        self.state_dict()['netlayers.6.weight'][:] = param_dict["L2_BatchNorm_W"]
        self.state_dict()['netlayers.6.bias'][:] = param_dict["L2_BatchNorm_b"]
        self.state_dict()["netlayers.9.weight"][:] = param_dict["L3_Linear_W"]
        self.state_dict()["netlayers.9.bias"][:] = param_dict["L3_Linear_b"].reshape(-1)
        return

    def get_net_parameters(self):
        """:returns a dictionary with the trainable parameters"""
        internal_dict = {name: value for name, value in self.named_parameters()}
        repr_dict = dict()
        repr_dict["L1_Linear_W"] = internal_dict["netlayers.1.weight"]
        repr_dict["L1_Linear_b"] = internal_dict["netlayers.1.bias"].reshape(-1, 1)
        repr_dict["L1_BatchNorm_W"] = internal_dict["netlayers.2.weight"]
        repr_dict["L1_BatchNorm_b"] = internal_dict["netlayers.2.bias"]
        repr_dict["L2_Linear_W"] = internal_dict["netlayers.5.weight"]
        repr_dict["L2_Linear_b"] = internal_dict["netlayers.5.bias"].reshape(-1, 1)
        repr_dict["L2_BatchNorm_W"] = internal_dict["netlayers.6.weight"]
        repr_dict["L2_BatchNorm_b"] = internal_dict["netlayers.6.bias"]
        repr_dict["L3_Linear_W"] = internal_dict["netlayers.9.weight"]
        repr_dict["L3_Linear_b"] = internal_dict["netlayers.9.bias"].reshape(-1, 1)
        return repr_dict



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

    net = MnistPiReal()
    net = net.to(device)

    num_epochs = 200
    # will save metrics and model to disk
    net.train_model(train_loader, validation_loader, test_loader, num_epochs, "real")

if __name__ == "__main__":
    print('Using device:', device)
    train_model()

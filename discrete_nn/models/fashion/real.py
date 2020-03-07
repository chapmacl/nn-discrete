from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import DataLoader
from discrete_nn.models.base_model import BaseModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device == "cuda:0":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class FashionReal(BaseModel):
    def __init__(self, weights=None):
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
        x = x.to(self.device)
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


def train_model():
    # creates the dataloader for pytorch
    batch_size = 100
    ToTensorMethod = ToTensor()

    def flatten_image(pil_image):
        return ToTensorMethod(pil_image).reshape(-1).to(device)

    def transform_target(target):
        return torch.tensor(target).to(device)

    from discrete_nn.settings import dataset_path
    import os
    mnist_fashion_path = os.path.join(dataset_path, "fashion")

    train_val_dataset = FashionMNIST(mnist_fashion_path, download=True, train=True, transform=flatten_image,
                                     target_transform=transform_target)

    train_size = int(len(train_val_dataset) * 0.8)
    eval_size = len(train_val_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, eval_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    test_loader = DataLoader(FashionMNIST(mnist_fashion_path, download=True, train=False, transform=flatten_image,
                                          target_transform=transform_target),
                             batch_size=batch_size)

    net = FashionReal()
    net = net.to(device)

    num_epochs = 100
    # will save metrics and model to disk
    net.train_model(train_loader, validation_loader, test_loader, num_epochs, "FashionMnistReal")


if __name__ == "__main__":
    print('Using device:', device)
    train_model()

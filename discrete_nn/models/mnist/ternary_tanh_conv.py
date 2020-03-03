"""
This module implements the real valued (non convolutionary) network for the mnist dataset
"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import os
import pickle
from sklearn.metrics import accuracy_score

from discrete_nn.dataset.mnist import MNIST
from discrete_nn.settings import model_path
from discrete_nn.layers.type_defs import ValueTypes, DiscreteWeights
from discrete_nn.layers.logit_linear import TernaryLinear
from discrete_nn.layers.conv import LogitConv
from discrete_nn.layers.local_reparametrization import LocalReparametrization
from discrete_nn.models.mnist.real_conv import MnistReal
from discrete_nn.layers.Flatten import Flatten

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device == "cuda:0":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class MnistTernaryTanh(torch.nn.Module):
    """
    Real valued (non convolutionary) network for the mnist dataset
    """

    def __init__(self, real_model_params):
        """

        :param real_model_params: a dictionary containing the real weights of the pretrained model
        """
        super().__init__()
        self.netlayers = torch.nn.Sequential(
            LogitConv(1, ValueTypes.REAL, 32, 5, 1, real_model_params["L1_Conv_W"], None, DiscreteWeights.TERNARY),
            #torch.nn.Conv2d(1, 32, 5, stride=1, bias=False),
            LocalReparametrization(),
            torch.nn.BatchNorm2d(32, momentum=0.1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(2),
            #
            torch.nn.Dropout(p=0.2),
            # torch.nn.Conv2d(32, 64, 5, stride=1, bias=False),
            LogitConv(32, ValueTypes.REAL, 64, 5, 1, real_model_params["L2_Conv_W"], None, DiscreteWeights.TERNARY),
            LocalReparametrization(),
            torch.nn.BatchNorm2d(64, momentum=0.1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(2),

            #
            Flatten(),
            torch.nn.Dropout(p=0.3),
            TernaryLinear(1024, ValueTypes.REAL, 512, real_model_params["L3_Linear_W"],
                          real_model_params["L3_Linear_b"], normalize_activations=False),
            LocalReparametrization(),
            torch.nn.BatchNorm1d(512, momentum=0.1),
            torch.nn.Tanh(),
            #
            TernaryLinear(512, ValueTypes.REAL, 10, real_model_params["L4_Linear_W"], real_model_params["L4_Linear_b"],
                          normalize_activations=True),
            LocalReparametrization(),
        )

        #setting weihts of torch layers




    def forward(self, x):
        # takes image vector
        return self.netlayers(x)


class DatasetMNIST(Dataset):
    """
    Dataset for pytorch's DataLoader
    """

    def __init__(self, x, y):
        self.x = torch.from_numpy(x) * 2 - 1
        self.y = torch.from_numpy(y).long()
        self.x = self.x.reshape((self.x.shape[0], 1, 28, 28))
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

    with open(model_path + "/mnist_conv_real.pickle", "rb") as f:
        real_model = pickle.load(f)
        real_param = real_model.get_net_parameters()
    net = MnistPiTernaryTanh(real_param)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    loss_fc = torch.nn.CrossEntropyLoss()
    # todo check regularization

    num_epochs = 200

    epochs_train_error = []
    epochs_validation_error = []

    for epoch_in in tqdm.tqdm(range(num_epochs), desc="epoch"):
        net.train()
        batch_loss_train = []
        # training part of epoch
        for batch_inx, (X, Y) in tqdm.tqdm(enumerate(train_loader), desc="batch"):
            optimizer.zero_grad()  # reset gradients from previous iteration
            # do forward pass
            net_output = net(X)
            # compute loss
            loss = loss_fc(net_output, Y)
            # backward propagate loss
            loss.backward()
            optimizer.step()
            tqdm.tqdm.write(f"loss {loss}")
            batch_loss_train.append(loss.item())
        epochs_train_error.append(np.mean(batch_loss_train))

        # starting epochs evaluation
        net.eval()
        validation_losses = []

        # disables gradient calculation since it is not needed
        with torch.no_grad():
            for batch_inx, (X, Y) in enumerate(validation_loader):
                outputs = net(X)
                loss = loss_fc(outputs, Y)
                validation_losses.append(loss)
        epochs_validation_error.append(torch.mean(torch.stack(validation_losses)))
        print(f"epoch {epoch_in + 1}/{num_epochs} "
              f"train loss: {epochs_train_error[-1]:.4f} / "
              f"validation loss: {epochs_validation_error[-1]:.4f}")

    with open(os.path.join(model_path, "mnist_pi_ternary_tanh.pickle"), "wb") as f:
        pickle.dump(net, f)

    # test network
    test_losses = []
    targets = []
    predictions = []
    # disables gradient calculation since it is not needed
    with torch.no_grad():
        for batch_inx, (X, Y) in enumerate(test_loader):
            outputs = net(X)
            loss = loss_fc(outputs, Y)
            test_losses.append(loss.item())

            output_probs = torch.nn.functional.softmax(outputs, dim=1)
            output_labels = output_probs.argmax(dim=1)
            predictions += output_labels.tolist()
            targets += Y.tolist()

    print(f"test accuracy : {accuracy_score(targets, predictions)}")
    print(f"test cross entropy loss:{np.mean(test_losses)}")


if __name__ == "__main__":
    print('Using device:', device)
    train_model()

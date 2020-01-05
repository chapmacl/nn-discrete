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
from discrete_nn.layers.types import ValueTypes
from discrete_nn.layers.linear import TernaryLinear
from discrete_nn.layers.local_reparametrization import LocalReparametrization

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device == "cuda:0":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class MnistPiTernaryTanh(torch.nn.Module):
    """
    Real valued (non convolutionary) network for the mnist dataset
    """

    def __init__(self, real_model_params):
        """

        :param real_model_params: a dictionary containing the real weights of the pretrained model
        """
        super().__init__()
        s1_l1_dropout = torch.nn.Dropout(p=0.2)
        s2_l1_linear = TernaryLinear(784, ValueTypes.REAL, 1200, real_model_params["L1_Linear_W"],
                                     real_model_params["L1_Linear_b"])
        s3_l1_repar = LocalReparametrization(1200, ValueTypes.GAUSSIAN) # outputs a value and not a dist.
        s4_l1_batchnorm = torch.nn.BatchNorm1d(1200, momentum=0.1)
        s5_l1_tanh = torch.nn.Tanh()

        s6_l2_dropout = torch.nn.Dropout(p=0.4)
        s7_l2_linear = TernaryLinear(1200, ValueTypes.REAL, 1200, real_model_params["L2_Linear_W"],
                                     real_model_params["L2_Linear_W"])
        s8_l2_repar = LocalReparametrization(1200, ValueTypes.GAUSSIAN)  # outputs a value and not a dist.
        s9_l2_batchnorm = torch.nn.BatchNorm1d(1200, momentum=0.1)
        s10_l2_tanh = torch.nn.Tanh()

        s6_l3_dropout = torch.nn.Dropout(p=0.4)
        s7_l3_linear = TernaryLinear(1200, ValueTypes.REAL, 10, real_model_params["L3_Linear_W"],
                                     real_model_params["L3_Linear_W"])
        s8_l3_repar = LocalReparametrization(10, ValueTypes.GAUSSIAN)
        # defining all the network's layers
        self.netlayers = torch.nn.Sequential(
            s1_l1_dropout,
            s2_l1_linear,
            s3_l1_repar,
            s4_l1_batchnorm,
            s5_l1_tanh,
            s6_l2_dropout,
            s7_l2_linear,
            s8_l2_repar,
            s9_l2_batchnorm,
            s10_l2_tanh,
            s6_l3_dropout,
            s7_l3_linear,
            s8_l3_repar)  # last layer outputs the unnormalized loglikelihood used by the softmax later

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

    with open(model_path+"/mnist_real_param.pickle", "rb") as f:
        real_param = pickle.load(f)

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

        with open(os.path.join(model_path, "mnist_pi_real.pickle"), "wb") as f:
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

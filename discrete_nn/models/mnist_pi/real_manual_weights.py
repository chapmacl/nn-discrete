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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device == "cuda:0":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class MnistPiReal(torch.nn.Module):
    """
    Real valued (non convolutionary) network for the mnist dataset
    """

    def __init__(self):
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

    def forward(self, x):
        # takes image vector
        return self.netlayers(x)

    def training_parameters(self):
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
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    loss_fc = torch.nn.CrossEntropyLoss()
    # todo check regularization

    num_epochs = 2

    epochs_train_error = []
    epochs_validation_error = []

    for epoch_in in range(num_epochs):
        net.train()
        batch_loss_train = []
        # training part of epoch
        for batch_inx, (X, Y) in enumerate(train_loader):
            optimizer.zero_grad()  # reset gradients from previous iteration
            # do forward pass
            net_output = net(X)
            # compute loss
            loss = loss_fc(net_output, Y)
            # backward propagate loss
            loss.backward()
            optimizer.step()
            batch_loss_train.append(loss.item())
        epochs_train_error.append(np.mean(batch_loss_train))

        # starting epochs evaluation
        net.eval()
        validation_losses = []
        targets = []
        predictions = []
        # disables gradient calculation since it is not needed
        with torch.no_grad():
            for batch_inx, (X, Y) in enumerate(validation_loader):
                outputs = net(X)
                loss = loss_fc(outputs, Y)
                validation_losses.append(loss)

                output_probs = torch.nn.functional.softmax(outputs, dim=1)
                output_labels = output_probs.argmax(dim=1)
                predictions += output_labels.tolist()
                targets += Y.tolist()

        epochs_validation_error.append(torch.mean(torch.stack(validation_losses)))
        print(f"epoch {epoch_in + 1}/{num_epochs} "
              f"train loss: {epochs_train_error[-1]:.4f} / "
              f"validation loss: {epochs_validation_error[-1]:.4f}"
              f"validation acc: {accuracy_score(targets, predictions)}")

        """with open(os.path.join(model_path, "mnist_pi_manual.pickle"), "wb") as f:
            pickle.dump(net, f)"""



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

    # Manually adjust the weights
    modules = net._modules["netlayers"]._modules
    layerIndices = ["1", "5", "9"]
    for mod in layerIndices:
        layer = modules[str(mod)]
        if hasattr(layer, "weight"):
            data = layer.weight
            min = torch.min(data)
            max = torch.max(data)
            for x in range(0, len(data)):
                normalized = 0.2 * ((data[x] - min) / (max - min)) - 0.1
                """if normalized.ndim == 0:
                    print("problem")
                for y in range(0, len(normalized)):
                    number = normalized[y]
                    if number < -0.333:
                        normalized[y] = -1
                    elif number < 0.333:
                        normalized[y] = 0
                    else:
                        normalized[y] = 1"""

                data[x] = normalized
            layer.weight.data = data

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

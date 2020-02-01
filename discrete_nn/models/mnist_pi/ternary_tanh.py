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
from discrete_nn.layers.logit_linear import TernaryLinear
from discrete_nn.layers.local_reparametrization import LocalReparametrization
from discrete_nn.models.mnist_pi.real import MnistPiReal

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
        s1_l1_dropout = torch.nn.Dropout(p=0.1)
        s2_l1_linear = TernaryLinear(784, ValueTypes.REAL, 1200, real_model_params["L1_Linear_W"],
                                     real_model_params["L1_Linear_b"])
        s3_l1_repar = LocalReparametrization(1200, ValueTypes.GAUSSIAN) # outputs a value and not a dist.
        s4_l1_batchnorm = torch.nn.BatchNorm1d(1200, momentum=0.1)
        s5_l1_tanh = torch.nn.Tanh()

        s6_l2_dropout = torch.nn.Dropout(p=0.2)
        s7_l2_linear = TernaryLinear(1200, ValueTypes.REAL, 1200, real_model_params["L2_Linear_W"],
                                     real_model_params["L2_Linear_W"])
        s8_l2_repar = LocalReparametrization(1200, ValueTypes.GAUSSIAN)  # outputs a value and not a dist.
        s9_l2_batchnorm = torch.nn.BatchNorm1d(1200, momentum=0.1)
        s10_l2_tanh = torch.nn.Tanh()

        s6_l3_dropout = torch.nn.Dropout(p=0.3)
        s7_l3_linear = TernaryLinear(1200, ValueTypes.REAL, 10, real_model_params["L3_Linear_W"],
                                     real_model_params["L3_Linear_W"], normalize_activations=True)
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

        """setting parameters of torch layers here"""
        self.state_dict()['netlayers.3.weight'][:] = real_model_params["L1_BatchNorm_W"]
        self.state_dict()['netlayers.3.bias'][:] = real_model_params["L1_BatchNorm_b"]
        self.state_dict()['netlayers.8.weight'][:] = real_model_params["L2_BatchNorm_W"]
        self.state_dict()['netlayers.8.bias'][:] = real_model_params["L2_BatchNorm_b"]
        self.loss_fc = torch.nn.CrossEntropyLoss()
    def forward(self, x):
        # takes image vector
        return self.netlayers(x)

    def generate_discrete_networks(self, method: str) -> MnistPiReal:
        """

        :param method: sample or argmax
        :return:
        """
        # state dicts

        l1_layer: TernaryLinear = self.netlayers[1]
        l1_sampled_w, l1_sampled_b = l1_layer.generate_discrete_network(method)
        l2_layer: TernaryLinear = self.netlayers[5]
        l2_sampled_w, l2_sampled_b = l2_layer.generate_discrete_network(method)
        l3_layer: TernaryLinear = self.netlayers[9]
        l3_sampled_w, l3_sampled_b = l3_layer.generate_discrete_network(method)
        state_dict = {
            "L1_Linear_W": l1_sampled_w,
            "L1_Linear_b": l1_sampled_b,
            "L1_BatchNorm_W": self.state_dict()['netlayers.3.weight'],
            "L1_BatchNorm_b": self.state_dict()['netlayers.3.bias'],
            "L2_Linear_W": l2_sampled_w,
            "L2_Linear_b": l2_sampled_b,
            "L2_BatchNorm_W": self.state_dict()['netlayers.8.weight'],
            "L2_BatchNorm_b": self.state_dict()['netlayers.8.bias'],
            "L3_Linear_W": l3_sampled_w,
            "L3_Linear_b": l3_sampled_b
        }

        real_net = MnistPiReal()
        real_net.set_training_parameters(state_dict)
        return real_net

    def evaluate(self, dataset_generator):
        self.eval()
        validation_losses = []
        targets = []
        predictions = []
        # disables gradient calculation since it is not needed
        with torch.no_grad():
            for batch_inx, (X, Y) in enumerate(dataset_generator):
                outputs = self(X)
                loss = self.loss_fc(outputs, Y)
                validation_losses.append(loss)

                output_probs = torch.nn.functional.softmax(outputs, dim=1)
                output_labels = output_probs.argmax(dim=1)
                predictions += output_labels.tolist()
                targets += Y.tolist()
        eval_loss = torch.mean(torch.stack(validation_losses))
        eval_acc = accuracy_score(targets, predictions)
        return eval_loss, eval_acc

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


def train_logit_model(train_loader, validation_loader, test_loader, num_epochs, real_param) -> MnistPiTernaryTanh:
    net = MnistPiTernaryTanh(real_param)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-8)
    epochs_train_error = []
    epochs_validation_error = []

    for epoch_in in tqdm.tqdm(range(num_epochs), desc="epoch"):
        net.train()
        batch_loss_train = []
        # training part of epoch
        for batch_inx, (X, Y) in enumerate(train_loader):
            optimizer.zero_grad()  # reset gradients from previous iteration
            # do forward pass
            net_output = net(X)
            # compute loss
            loss = net.loss_fc(net_output, Y)
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
                loss = net.loss_fc(outputs, Y)
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
    return net

if __name__ == "__main__":
    batch_size = 100
    # basic dataset holder
    mnist = MNIST()
    # creates the dataloader for pytorch

    train_loader = DataLoader(dataset=DatasetMNIST(mnist.x_train, mnist.y_train), batch_size=batch_size,
                              shuffle=True)
    validation_loader = DataLoader(dataset=DatasetMNIST(mnist.x_val, mnist.y_val), batch_size=batch_size,
                                   shuffle=False)
    test_loader = DataLoader(dataset=DatasetMNIST(mnist.x_test, mnist.y_test), batch_size=batch_size,
                             shuffle=False)

    print('Using device:', device)
    with open(model_path+"/mnist_real_param.pickle", "rb") as f:
        real_param = pickle.load(f)
    num_epochs_logits = 5
    logit_net = train_logit_model(train_loader, validation_loader, test_loader, num_epochs_logits, real_param)

    #with open(model_path+"/mnist_pi_ternary_tanh.pickle", "rb") as f:
    #    logit_net: MnistPiTernaryTanh = pickle.load(f)

    # discretizing and evaluatin
    discrete_net = logit_net.generate_discrete_networks("sample")
    test_loss, test_acc = discrete_net.evaluate(test_loader)
    print(f"test_loss: {test_loss}, test_acc; {test_acc}")
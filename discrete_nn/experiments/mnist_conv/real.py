from torch.utils.data import DataLoader
import torch

from discrete_nn.dataset.mnist import MNIST
from discrete_nn.models.conv.real import ConvReal


def train_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # basic dataset holder
    mnist = MNIST(device, "2d")
    # creates the dataloader for pytorch
    batch_size = 100

    train_loader = DataLoader(dataset=mnist.train, batch_size=batch_size,
                              shuffle=True)
    validation_loader = DataLoader(dataset=mnist.validation, batch_size=batch_size,
                                   shuffle=False)
    test_loader = DataLoader(dataset=mnist.test, batch_size=batch_size,
                             shuffle=False)

    net = ConvReal()
    net = net.to(device)

    num_epochs = 100
    # will save metrics and model to disk. returns the path to metrics and saved model
    return net.train_model(train_loader, validation_loader, test_loader, num_epochs, model_name="MNIST-conv-real")


if __name__ == "__main__":
    train_model()

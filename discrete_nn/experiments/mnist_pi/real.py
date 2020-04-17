import torch
from torch.utils.data import DataLoader

from discrete_nn.dataset.mnist import MNIST
from discrete_nn.models.pi.real import PiReal


def train_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # basic dataset holder
    mnist = MNIST(device, "flat")
    # creates the dataloader for pytorch
    batch_size = 100
    train_loader = DataLoader(dataset=mnist.train, batch_size=batch_size,
                              shuffle=True)
    validation_loader = DataLoader(dataset=mnist.validation, batch_size=batch_size,
                                   shuffle=False)
    test_loader = DataLoader(dataset=mnist.test, batch_size=batch_size,
                             shuffle=False)

    net = PiReal()
    net = net.to(device)

    num_epochs = 100
    # will save metrics and model to disk
    return net.train_model(train_loader, validation_loader, test_loader, num_epochs, model_name="MNIST-pi-real")


if __name__ == "__main__":
    train_model()

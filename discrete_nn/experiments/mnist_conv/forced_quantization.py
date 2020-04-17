import torch
from torch.utils.data import DataLoader
from discrete_nn.dataset.mnist import MNIST
from discrete_nn.models.conv.forced_quantization import ConvForcedQuantization


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

    net = ConvForcedQuantization()
    net = net.to(device)
    num_epochs = 200

    net.train_model(train_loader, validation_loader, test_loader, num_epochs, "MNIST-Conv-Forced-Quantization")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    train_model()

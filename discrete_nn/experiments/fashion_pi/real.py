import torch

from discrete_nn.dataset.fashion import fashion_mnist_dataloaders
from discrete_nn.models.pi.real import PiReal


def train_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # basic dataset holder
    # creates the dataloader for pytorch
    batch_size = 100
    train_loader, validation_loader, test_loader = fashion_mnist_dataloaders(batch_size, device, "flat")
    net = PiReal()
    net = net.to(device)

    num_epochs = 500
    # will save metrics and model to disk
    return net.train_model(train_loader, validation_loader, test_loader, num_epochs, model_name="Fashion-pi-real-500epoch")


if __name__ == "__main__":
    train_model()

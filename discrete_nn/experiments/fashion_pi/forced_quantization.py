import torch
from torch.utils.data import DataLoader


from discrete_nn.dataset.fashion import fashion_mnist_dataloaders
from discrete_nn.models.pi.forced_quantization import PiForcedQuantization


def train_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # creates the dataloader for pytorch
    batch_size = 100
    train_loader, validation_loader, test_loader = fashion_mnist_dataloaders(batch_size, device, "flat")

    net = PiForcedQuantization()
    net = net.to(device)
    num_epochs = 200

    net.train_model(train_loader, validation_loader, test_loader, num_epochs, "Fashion-Pi-Forced-Quantization")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    train_model()

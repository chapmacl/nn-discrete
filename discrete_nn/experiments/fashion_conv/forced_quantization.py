import torch


from discrete_nn.dataset.fashion import fashion_mnist_dataloaders
from discrete_nn.models.conv.forced_quantization import ConvForcedQuantization


def train_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # creates the dataloader for pytorch
    batch_size = 100
    train_loader, validation_loader, test_loader = fashion_mnist_dataloaders(batch_size, device, "2d")

    net = ConvForcedQuantization()
    net = net.to(device)
    num_epochs = 200

    net.train_model(train_loader, validation_loader, test_loader, num_epochs, "Fashion-Conv-Forced-Quantization")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    train_model()

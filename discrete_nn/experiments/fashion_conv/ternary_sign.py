import os
import torch

from discrete_nn.dataset.fashion import fashion_mnist_dataloaders
from discrete_nn.settings import model_path
from discrete_nn.layers.type_defs import DiscreteWeights
from discrete_nn.models.conv.logit_sign import ConvLogitSign


def train_model(real_model_folder):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 100

    real_model_param_path = os.path.join(model_path, real_model_folder,
                                         "PiReal.param.pickle")

    real_param = torch.load(real_model_param_path, map_location="cpu")
    logit_net = ConvLogitSign(real_param, DiscreteWeights.TERNARY)
    logit_net = logit_net.to(device)
    train_loader, validation_loader, test_loader = fashion_mnist_dataloaders(batch_size, device, "2d")

    logit_net.train_model(train_loader, validation_loader, test_loader, 200, "Fashion-Conv-Sign-Ternary", True)


if __name__ == "__main__":
    train_model("MNIST-pi-real-trained-2020-3-8--h17m34")

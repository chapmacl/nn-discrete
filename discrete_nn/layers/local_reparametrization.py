from discrete_nn.layers.type_defs import ValueTypes

from torch import nn
from torch import Tensor
from torch.distributions.normal import Normal
import torch


class LocalReparametrization(nn.Module):
    def __init__(self):
        super().__init__()
        self.sampling_epsilon = 1e-6

    def forward(self, x):
        """Applies local reparametrization trick to input
        :param x: a tensor with dimensions (n x 2 x num_features) or (n x 2 x out channels x num rows image x num col images) where the second
        dimension contains, in order, the mean and variance,  n is the number of samples and, num_features
         is the number of out neurons."""

        if self.input_type == ValueTypes.GAUSSIAN:
            # sample from the normal distributions
            if x.ndim == 3: # a 1d feature vector
                x_mean = x[:, 0, :]
                x_var = x[:, 1, :]
            elif x.ndim == 5:  # a convolutional output
                x_mean = x[:, 0, :, :, :]
                x_var = x[:, 1, :, :, :]
            else:
                raise ValueError(f"x has invalid shape : {x.shape}")
            ndist = Normal(torch.tensor([0.]), torch.tensor([1.]))
            samples_unit_gaussian = ndist.sample(sample_shape=x_mean.shape).to(x.device)
            # an extra dimension is added when samples are generated we need to remove it
            if x.ndim == 3:  # a 1d feature vector
                samples_unit_gaussian = samples_unit_gaussian[:, :, 0]
            elif x.ndim == 5:  # a convolutional output
                samples_unit_gaussian = samples_unit_gaussian[:, :, :, :, 0]
            else:
                raise ValueError(f"x has invalid shape : {x.shape}")
            x_out = x_mean + torch.sqrt(x_var+self.sampling_epsilon)*samples_unit_gaussian
            return x_out.float()
        else:
            raise ValueError(f"unsupported input type {self.input_type}")


if __name__ == "__main__":
    lr = LocalReparametrization()
    input_tensor = torch.tensor([[[0.0, 0.4, 1.5, 3.2], [1.1, 3, 2., 1.]]])
    print(input_tensor.shape)
    print(input_tensor)
    print(lr.forward(input_tensor))
    print(lr.forward(input_tensor).type())

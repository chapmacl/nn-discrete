from discrete_nn.layers.types import ValueTypes

from torch import nn
from torch import Tensor
from torch.distributions.normal import Normal
import torch


class LocalReparametrization(nn.Module):
    def __init__(self, num_features, input_feature_type: ValueTypes):
        super().__init__()
        self.num_neurons = num_features
        self.input_type = input_feature_type
        self.sampling_epsilon = 1e-6

    def forward(self, x):
        """Applies local reparametrization trick to input
        :param x: a tensor with dimensions (n x 2 x num_features) where the second
        dimension contains, in order, the mean and variance,  n is the number of samples and, num_features
         is the number of out neurons."""

        if self.input_type == ValueTypes.GAUSSIAN:
            # sample from the normal distributions
            x_mean = x[:, 0, :]
            x_var = x[:, 1, :]
            ndist = Normal(torch.tensor([0.]), torch.tensor([1.]))
            samples_unit_gaussian = ndist.sample(sample_shape=x_mean.shape)
            # an extra dimension is added when samples are generated we need to remove it
            samples_unit_gaussian = samples_unit_gaussian[:, :, 0]

            x_out = x_mean + torch.sqrt(x_var+self.sampling_epsilon)*samples_unit_gaussian
            return x_out.float()
        else:
            raise ValueError(f"unsupported input type {self.input_type}")


if __name__ == "__main__":
    lr = LocalReparametrization(4, ValueTypes.GAUSSIAN)
    input_tensor = torch.tensor([[[0.0, 0.4, 1.5, 3.2], [1.1, 3, 2., 1.]]])
    print(input_tensor.shape)
    print(input_tensor)
    print(lr.forward(input_tensor))
    print(lr.forward(input_tensor).type())

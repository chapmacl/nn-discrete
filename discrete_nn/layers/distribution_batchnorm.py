from torch.nn import Module
import torch

from discrete_nn.layers.type_defs import InputFormat


class DistributionBatchnorm(Module):
    """
    Implements batch normalization over distributions for 1d and 2d scenarios. Only supports distributions as inputs.
    """

    def __init__(self, input_format: InputFormat, num_features, initialization_weights, initialization_bias):
        """
        Initialized a batch normalization layer
        :param input_format: defines if input is a feature map or an array, both of which defined with distributions
        :param num_features: the number of features in input. In the case of a feature map, corresponds to the number
        :param initialization_weights: the weight with which the layer should be initialized with
        :param initialization_weights: the bias with which the layer should be initialized with

        of channels
        """
        super().__init__()
        self._input_format = input_format
        self._num_feat = num_features
        self.gamma = torch.nn.Parameter(initialization_weights, requires_grad=True)
        self.beta = torch.nn.Parameter(initialization_bias, requires_grad=True)

    def forward(self, x):
        batch_size = x.shape[0]
        if self._input_format == InputFormat.FEATURE_MAP:

            # its a distribution feature map of the shape (batch size x 2 x channels x h x w)
            input_mean = x[:, 0, :, :, :]  # (batch size x channels x h x w)
            input_var = x[:, 1, :, :, :]
        elif self._input_format == InputFormat.FLAT_ARRAY:
            # its a flat input with distributions
            input_mean = x[:, 0, :]  # a (batch_size, num_inputs) tensor
            input_var = x[:, 1, :]
        else:
            raise ValueError(f"invalid shape for input:{x.shape}")

        # a tensor with shape (num inputs) if input is a flat array, (channels x h x w) if a feature map
        batch_mean = torch.sum(input_mean, dim=0) / batch_size
        batch_var = torch.sum(input_var + torch.pow(input_mean - batch_mean, 2), dim=0) / (batch_size - 1)

        normalized_mean = (input_mean - batch_mean)/torch.sqrt(batch_var)
        normalized_var = input_var/batch_var
        # to avoid having gamma be multiplied over the wrong dimension in case one of the dimensions has the same size
        # as the features we move the feature dimension to the innermost position for multiplication when dealing
        # with feature maps.
        if self._input_format == InputFormat.FEATURE_MAP:
            normalized_mean = normalized_mean.transpose(1, 2).transpose(2, 3)
            normalized_var = normalized_var.transpose(1, 2).transpose(2, 3)

        out_mean = self.gamma * normalized_mean + self.beta
        out_var = normalized_var * torch.pow(self.gamma, 2)

        if self._input_format == InputFormat.FEATURE_MAP:
            # returns channels dimension to its proper place
            out_mean = out_mean.transpose(3, 2).transpose(2, 1)
            out_var = out_var.transpose(3, 2).transpose(2, 1)
        return torch.stack([out_mean, out_var], dim=1)


if __name__ == "__main__":
    num_channels = 4
    b = DistributionBatchnorm(InputFormat.FEATURE_MAP, num_channels)
    h = 10
    w = 11
    batch_s = 100
    input_tensor = torch.rand(batch_s, 2, num_channels, h, w)
    b.forward(input_tensor)
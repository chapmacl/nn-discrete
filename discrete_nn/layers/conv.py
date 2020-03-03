from torch import nn
from torch import Tensor
from torch.distributions import Multinomial
import torch

from discrete_nn.layers.types import ValueTypes
from discrete_nn.layers.weight_utils import discretize_weights_probabilistic

class LogitLinear(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, kernel_size, stride, initialization_weights,
                 initialization_bias, discrete_weight_values, normalize_activations: bool = False):
        """
        Initializes a logit convolutional layer
        :param num_input_channels: number of input channels
        :param num_output_channels: ... of output channels
        :param kernel_size: a integer for a square kernel or a 2d tuple with the kernel size
        :param stride: a integer or 2d tuple for stride
        :param initialization_weights: a tensor (output_channels x input_channels x kernel_rows x kernel_columns) with pre trained real weights to b/e
        used for initializing the discrete ones
        :param initialization_bias: a tensor (output_channels) with pre trained bias to be
        used for initializing the discrete ones
        :param discrete_weight_values: a list with possible discrete weight values
        :param normalize_activations:
        """
        if type(kernel_size) == int and kernel_size > 0:
            self.kernel_size = (kernel_size, kernel_size)
        elif type(kernel_size) == tuple and len(kernel_size) == 2 and type(kernel_size[0])==int and \
                type(kernel_size[1]) == int and kernel_size[0] > 0 and kernel_size[1] > 0:
            self.kernel_size = kernel_size
        else:
            raise ValueError(f"provided kernel size ({kernel_size}) is invalid.")

        if type(stride) == int and stride > 0:
            self.stride = (stride, stride)
        elif type(stride) == tuple and len(stride) == 2 and type(stride[0]) == int and \
                type(stride[1]) == int and stride[0] > 0 and stride[1] > 0:
            self.stride = stride
        else:
            raise ValueError(f"provided stride ({stride}) is invalid.")

        self.n_input_channels = num_input_channels
        self.n_output_channels = num_output_channels
        self.discrete_weight_values = discrete_weight_values

        if initialization_weights.shape != (self.n_output_channels, self.n_input_channels,
                                            self.kernel_size[0], self.kernel_size[1]):
            raise ValueError(f"initialization weights of invalid shape. Expected: "
                             f"{(self.n_output_channels, self.n_input_channels, self.kernel_size[0], self.kernel_size[1])}"
                             f" but got: {initialization_weights.shape}")
        if initialization_bias.shape != (self.n_output_channels):
            raise ValueError(f"initialization weights of invalid shape. Expected: "
                             f"{(self.n_output_channels)}"
                             f" but got: {initialization_bias.shape}")
        # generating multinomial distribution over the kernel weights
        # need to flatten the kernel weights
        flat_weights = initialization_weights.view(self.n_output_channels, self.n_input_channels, -1)
        # returns a (discrete_values x output_channels x input_channels x -1) matrix
        flat_weight_distribution = discretize_weights_probabilistic(flat_weights, self.discrete_weight_values)
        # unflatten the kernel weights and set as parameters

        # (discrete_values x output_channels x input_channels x kernel_rows x kernel_columns)
        self.W_logits = torch.nn.Parameter(flat_weight_distribution.view(len(self.discrete_weight_values),
                                                                         self.n_output_channels,
                                                                         self.n_input_channels, self.kernel_size[0],
                                                                         self.kernel_size[1]),
                                           requires_grad=True)

        # this is a (n_output_channels x 1) tensor
        self.b_logits = torch.nn.Parameter(
            discretize_weights_probabilistic(initialization_bias.view(self.n_output_channels, 1), self.discrete_values),
                                           requires_grad=True)


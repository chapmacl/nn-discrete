from typing import Optional
from torch.nn import Module
import torch
from discrete_nn.layers.weight_utils import discretize_weights_probabilistic, get_gaussian_dist_parameters
from discrete_nn.layers.type_defs import InputFormat, ValueTypes


class DistributionBatchnorm(Module):
    """
    Implements batch normalization with inputs that are distributions for 1d and 2d scenarios.
    Only supports distributions as inputs. Its weights and bias are real
    """

    def __init__(self, input_format: InputFormat, value_type: ValueTypes, num_features,
                 initialization_weights: Optional[torch.tensor],
                 initialization_bias: Optional[torch.tensor]):
        """
        Initialized a batch normalization layer
        :param input_format: defines if input is a feature map or an array, both of which defined with distributions
        :param value_type: defines the type of input AND output value
        :param num_features: the number of features in input. In the case of a feature map, corresponds to the number
        :param initialization_weights: the weight with which the layer should be initialized with
        :param initialization_weights: the bias with which the layer should be initialized with

        of channels
        """
        super().__init__()
        self._input_format = input_format
        self._num_feat = num_features
        self._value_type = value_type
        if initialization_bias is None:
            initialization_bias = torch.zeros(num_features)
        if initialization_weights is None:
            initialization_weights = torch.ones(num_features)

        self.gamma = torch.nn.Parameter(initialization_weights, requires_grad=True)
        self.beta = torch.nn.Parameter(initialization_bias, requires_grad=True)

    def forward(self, x):
        batch_size = x.shape[0]
        if self._input_format == InputFormat.FEATURE_MAP:
            # each channel is a feature
            # to avoid having gamma being multiplied over the wrong dimension in case one of the dimensions has the same
            # size as the features (or a similar issue when we subtract the batch means) we move the feature dimension
            # to the innermost position for multiplication when dealing with feature maps.
            if self._value_type == ValueTypes.GAUSSIAN:
                # x its a distribution feature map of the shape (batch size x 2 x channels x h x w)
                shifted_x = x.transpose(2, 3).transpose(3, 4)
                # shifted x is of the shape (batch_size x 2 x h x w x channels)
                input_mean = shifted_x[:, 0, :, :, :]  # (batch size  x h x w x channels)
                input_var = shifted_x[:, 1, :, :, :]
                h = shifted_x.shape[2]
                w = shifted_x.shape[3]

            elif self._value_type == ValueTypes.REAL:
                # x its a distribution feature map of the shape (batch size x channels x h x w)
                shifted_x = x.transpose(1, 2).transpose(2, 3)
                # shifted x is of the shape (batch_size x h x w x channels)
                input_mean = shifted_x[:, :, :, :]  # (batch size  x h x w x channels)
                input_var = 0.0
                h = shifted_x.shape[1]
                w = shifted_x.shape[2]
            else:
                raise TypeError(f"invalid ValueType:{self._value_type}")

            # a vector of size #channels
            batch_mean = torch.sum(input_mean, dim=[0, 1, 2]) / (batch_size * w * h)
            batch_var = torch.sum(input_var + torch.pow(input_mean - batch_mean, 2), dim=[0, 1, 2]) / (
                                  batch_size*h*w - 1)

        elif self._input_format == InputFormat.FLAT_ARRAY:

            if self._value_type == ValueTypes.GAUSSIAN:
                # its a flat input with distributions
                input_mean = x[:, 0, :]  # a (batch_size, num_inputs) tensor
                input_var = x[:, 1, :]
            elif self._value_type == ValueTypes.REAL:
                # its a flat input with distributions
                input_mean = x[:, :]  # a (batch_size, num_inputs) tensor
                input_var = 0.0
            else:
                raise TypeError(f"invalid ValueType:{self._value_type}")
            # vector corresponding to number of input features
            batch_mean = torch.sum(input_mean, dim=0) / batch_size
            batch_var = torch.sum(input_var + torch.pow(input_mean - batch_mean, 2), dim=0) / (batch_size - 1)
        else:
            raise ValueError(f"invalid shape for input:{x.shape}")

        assert batch_mean.shape == (self._num_feat,)
        normalized_mean = (input_mean - batch_mean)/torch.sqrt(batch_var + 1e-8)

        out_mean = self.gamma * normalized_mean + self.beta
        if self._value_type == ValueTypes.GAUSSIAN:
            normalized_var = input_var/batch_var
            out_var = normalized_var * torch.pow(self.gamma, 2)
        else:
            out_var = None

        if self._input_format == InputFormat.FEATURE_MAP:
            # out_mean and out_var are calculated from input mean
            # which is of the shape (batch size x h x w x channels)
            # we need to returns channels dimension to its proper place
            out_mean = out_mean.transpose(3, 2).transpose(2, 1)
            if self._value_type == ValueTypes.GAUSSIAN:
                out_var = out_var.transpose(3, 2).transpose(2, 1)

        if self._value_type == ValueTypes.GAUSSIAN:
            return torch.stack([out_mean, out_var], dim=1)
        else:
            return out_mean


if __name__ == "__main__":
    num_feat = 4
    weight_vector = torch.rand(num_feat)
    bias_vector = torch.rand(num_feat)
    bias_vector[0] = 100
    b = DistributionBatchnorm(InputFormat.FEATURE_MAP, num_feat, weight_vector, bias_vector)
    batch_s = 100
    input_tensor = torch.rand(batch_s, 2, num_feat, 10, 10)
    input_tensor[:,1,:,:,:] = 1
    dist_batch_norm = b.forward(input_tensor)
    from torch.nn import BatchNorm2d

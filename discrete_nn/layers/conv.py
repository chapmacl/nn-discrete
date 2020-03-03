from enum import Enum
from torch import nn
from torch import Tensor
from torch.distributions import Multinomial
import torch
import math

from typing import Union, Tuple

from discrete_nn.layers.type_defs import ValueTypes, DiscreteWeights
from discrete_nn.layers.weight_utils import discretize_weights_probabilistic


class LogitConv(nn.Module):
    def __init__(self, num_input_channels, input_feature_types: ValueTypes, num_output_channels,
                 kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], initialization_weights,
                 initialization_bias, discrete_weight_values: DiscreteWeights):
        """
        Initializes a logit convolutional layer
        :param num_input_channels: number of input channels
        :param num_output_channels: ... of output channels
        :param kernel_size: a integer for a square kernel or a 2d tuple with the kernel size
        :param stride: a integer or 2d tuple for stride
        :param initialization_weights: a tensor (output_channels x input_channels x kernel_rows x kernel_columns) with
         pre trained real weights to be used for initializing the discrete ones
        :param initialization_bias: a tensor (output_channels) with pre trained bias to be
        used for initializing the discrete ones
        :param discrete_weight_values: a value from the DiscreteWeights
        """
        super().__init__()
        if type(kernel_size) == int and kernel_size > 0:
            self.kernel_size = (kernel_size, kernel_size)
        elif type(kernel_size) == tuple and len(kernel_size) == 2 and type(kernel_size[0]) == int and \
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
        self.discrete_weight_values = discrete_weight_values.value

        if initialization_weights.shape != torch.Size([self.n_output_channels, self.n_input_channels,
                                                       self.kernel_size[0], self.kernel_size[1]]):
            raise ValueError(f"initialization weights of invalid shape. Expected: "
                             f"{(self.n_output_channels, self.n_input_channels, self.kernel_size[0], self.kernel_size[1])}"
                             f" but got: {initialization_weights.shape}")

        # generating multinomial distribution over the kernel weights
        # need to flatten the kernel weights
        flat_weights = initialization_weights.view(self.n_output_channels, self.n_input_channels, -1)
        # returns a (discrete_values x output_channels x input_channels x -1) matrix
        flat_weight_distribution = discretize_weights_probabilistic(flat_weights, self.discrete_weight_values)
        # unflatten the kernel weights and set as parameters

        self.W_logits = torch.nn.Parameter(flat_weight_distribution.view(len(self.discrete_weight_values),
                                                                         self.n_output_channels,
                                                                         self.n_input_channels, self.kernel_size[0],
                                                                         self.kernel_size[1]),
                                           requires_grad=True)
        """a (discrete_values x output_channels x input_channels x kernel_rows x kernel_columns)"""

        self.b_logits = None
        """this is a (discrete_values x n_output_channels x 1) tensor or None"""

        if initialization_bias is not None:
            if initialization_bias.shape != torch.Size([self.n_output_channels]):
                raise ValueError(f"initialization weights of invalid shape. Expected: "
                                 f"[{self.n_output_channels}]"
                                 f" but got: {initialization_bias.shape}")

            self.b_logits = torch.nn.Parameter(
                discretize_weights_probabilistic(initialization_bias.view(self.n_output_channels, 1), self.discrete_weight_values),
                requires_grad=True)
            """this is a (discrete_values x n_output_channels x 1) tensor or None"""

        # checks input
        self.input_type = ValueTypes(input_feature_types)

    @staticmethod
    def generate_weight_probabilities(logit_weights):
        """
        calculates the probabilities of all discrete weights for the logits provided

        :param logit_weights: a tensor with dimensions (discrete_values x output_channels x input_channels x
            kernel_rows x kernel_columns) with the discrete distribution as logits
        :return:  a tensor with dimensions same dimensions as input with the discrete distribution as probabilities
        """
        weight_probabilities = torch.exp(logit_weights)
        weight_probabilities = weight_probabilities / weight_probabilities.sum(dim=0)
        return weight_probabilities

    def get_gaussian_dist_parameters(self, logit_weights):
        """
        Fits a gaussian distribution to the logits in logit_weights.
        :param logit_weights: a tensor with dimensions (discrete_values x output_channels x input_channels x
            kernel_rows x kernel_columns) with the discrete distribution as logits if the distribution is over weights.
            (discrete_values x output_channels x 1) if its over bias
        :return: a tuple with the means of the gaussian distributions as a (output_channels x input_channels x
            kernel_rows x kernel_columns) / (output_channels x 1) tensor
            and a the standard deviations in a tensor of the same format for weights/bias
        """
        weight_probabilities = self.generate_weight_probabilities(logit_weights)
        discrete_val_tensor = torch.zeros_like(logit_weights)
        for inx, discrete_weight in enumerate(self.discrete_weight_values):
            discrete_val_tensor[inx] = discrete_weight
        discrete_val_tensor.requires_grad = True
        weight_mean = discrete_val_tensor * weight_probabilities
        weight_mean = weight_mean.sum(dim=0)

        weight_var = weight_probabilities * torch.pow(discrete_val_tensor - weight_mean, 2)
        weight_var = weight_var.sum(dim=0)
        return weight_mean, weight_var

    def generate_discrete_network(self, method: str = "sample"):
        """ generates discrete weights from the weights of the layer based on the weight distributions

        :param method: the method to use to generate the discrete weights. Either argmax or sample

        :returns: tuple (sampled_w, sampled_b) where sampled_w and sampled_b are tensors of the shapes
        (output_channels x input_channels x kernel rows x kernel columns) and (output_features x 1). sampled_b is None if the layer has no bias
        """

        probabilities_w = self.generate_weight_probabilities(self.W_logits)
        # logit probabilities must be in inner dimension for torch.distribution.Multinomial
        # stepped transpose bc we need to keep the order of the other dimensions
        probabilities_w = probabilities_w.transpose(0, 1).transpose(1, 2).transpose(2, 3).transpose(3, 4)
        if self.b_logits is not None:
            probabilities_b = self.generate_weight_probabilities(self.b_logits)
            probabilities_b = probabilities_b.transpose(0, 1).transpose(1, 2)
        else:
            # layer does not use bias
            probabilities_b = None
        discrete_values_tensor = torch.tensor(self.discrete_weight_values).double()
        discrete_values_tensor = discrete_values_tensor.to(self.W_logits.device)
        if method == "sample":
            # this is a output_channels x input_channels x kernel rows x kernel columns x discretization_levels mask
            m_w = Multinomial(probs=probabilities_w)
            sampled_w = m_w.sample()
            if torch.all(sampled_w.sum(dim=4) != 1):
                raise ValueError("sampled mask for weights does not sum to 1")

            # need to generate the discrete weights from the masks
            sampled_w = torch.matmul(sampled_w, discrete_values_tensor)

            if probabilities_b:
                # this is a output channels x 1 x discretization levels mask
                m_b = Multinomial(probs=probabilities_b)
                sampled_b = m_b.sample()

                if torch.all(sampled_b.sum(dim=2) != 1):
                    raise ValueError("sampled mask for bias does not sum to 1")
                sampled_b = torch.matmul(sampled_b, discrete_values_tensor)
            else:
                sampled_b = None

        elif method == "argmax":
            # returns a (out_feat x in_feat) matrix where the values correspond to the index of the discretized value
            # with the largest probability
            argmax_w = torch.argmax(probabilities_w, dim=4)
            # creating placeholder for discrete weights
            sampled_w = torch.zeros_like(argmax_w).to("cpu")
            sampled_w[:] = discrete_values_tensor[argmax_w[:]]

            if probabilities_b:
                argmax_b = torch.argmax(probabilities_b, dim=2)
                sampled_b = torch.zeros_like(argmax_b).to("cpu")
                sampled_b[:] = discrete_values_tensor[argmax_b[:]]
            else:
                sampled_b = None
        else:
            raise ValueError(f"Invalid method {method} for layer discretization")

        # sanity checks
        if sampled_w.shape != probabilities_w.shape[:-1]:
            raise ValueError("sampled probability mask for weights does not match expected shape")
        if sampled_b:
            if sampled_b.shape != probabilities_b.shape[-1]:
                raise ValueError("sampled probability mask for bias does not match expected shape")

        return sampled_w, sampled_b

    def forward(self, input_batch: torch.Tensor):
        """

        :param input_batch: a (sizebatch x in channels x num rows image x num col images) tensor if the values are real.
         (sizebatch x 2 x in channels x num rows image x num col images) if the values are a distribution  where the
          second dimension contains, in order, the mean and variance,  n is the number of samples and, in_feat is the
          number of input features.
        :return: tensor with dimensions (sizebatch x 2 x out channels x num rows image x num col images) where the
            second dimension contains, in order, the mean and variance,  n is the number of samples and, out_feat is
            the number of out features.

        """
        input_batch = input_batch.double()
        w_mean, w_var = self.get_gaussian_dist_parameters(self.W_logits)
        # calculates padding
        padding = math.floor(self.kernel_size[0]/2), math.floor(self.kernel_size[1]/2)

        if self.input_type == ValueTypes.REAL:

            out_mean = nn.functional.conv2d(input_batch, w_mean, stride=self.stride, padding=padding)
            out_var = nn.functional.conv2d(torch.pow(input_batch, 2), w_var, stride=self.stride, padding=padding)
            """
            x_in = self._pad(self._input_layer.getTrainOutput())
            x_out_mean = T.nnet.conv2d(x_in, self._W_mean, **self._conv_args)
            x_out_var = T.nnet.conv2d(T.sqr(x_in), self._W_var, **self._conv_args)

             if self._enable_bias:
            
            """
        elif self.input_type == ValueTypes.GAUSSIAN:
            in_mean = input_batch[:, 0, :, :, :]
            in_var = input_batch[:, 1, :, :, :]

            out_mean = nn.functional.conv2d(in_mean, w_mean, stride=self.stride, padding=padding)
            out_var = nn.functional.conv2d(torch.pow(in_mean, 2), w_var, stride=self.stride, padding=padding) + \
                    nn.functional.conv2d(in_var, torch.pow(w_mean, 2), stride=self.stride, padding=padding) + \
                    nn.functional.conv2d(in_var, w_var, stride=self.stride, padding=padding)

        else:
            raise ValueError(f"no forward implementation for feature type : {self.input_type}")

        if self.b_logits is not None:
            # adding bias
            b_mean, b_var = self.get_gaussian_dist_parameters(self.b_logits)
            for out_channel_inx in range(b_mean.shape[0]):
                out_mean[:, out_channel_inx, :, :] += b_mean[out_channel_inx, 0]  # broadcasting to all samples
                out_var[:, out_channel_inx, :, :] += b_var[out_channel_inx, 0]
            return torch.stack([out_mean, out_var], dim=1)


if __name__ == "__main__":
    test_weight_matrix = torch.rand((1, 1, 3, 3))*2 - 1
    test_bias_matrix = torch.tensor([2.2])
    test_samples = torch.rand(10, 1, 20, 20).double()  # 10 samples, 4 in dimensions
    layer = LogitConv(1, ValueTypes.REAL, 1, (3, 3), 1,  test_weight_matrix, test_bias_matrix, [-2, -1, 0, 1, 2])
    x = layer.forward(test_samples)
    print(x.shape)


"""
implements pooling layers for distributions
"""
from enum import Enum
from typing import Tuple, Union
import math

from torch import nn
import torch
from torch.distributions import Normal


class DistributionMaxPool(nn.Module):
    def __init__(self, stride: Union[int, Tuple[int, int]], epsilon: float = 1e-8):
        super().__init__()
        self._epsilon = epsilon

        if type(stride) == int and stride > 0:
            self.stride = (stride, stride)
        elif type(stride) == tuple and len(stride) == 2 and type(stride[0]) == int and \
                type(stride[1]) == int and stride[0] > 0 and stride[1] > 0:
            self.stride = stride
        else:
            raise ValueError(f"provided stride ({stride}) is invalid.")
        # kernel size is fixed at 2
        self.kernel_size = 2
        # reference implementation always uses zero padding makes it simpler to calculate max pool with distributions
        self._padding = 0

    def forward(self, x):
        """

        :param x: tensor with dimensions (size batch x 2 x input channels x num rows image x num col images) where
            the second dimension contains, in order, the mean and variance
        :return:
        """
        if len(x.shape) != 5:
            raise TypeError(f"input with shape {x.shape} is invalid. Expected (size batch x 2 x input channels x "
                            f"num rows image x num col images)")
        if x.shape[1] != 2:
            raise TypeError("input is not a distribution")

        return self._calculate_max_of_gaussians(x)


    def _gen_neighbours(self, input_tensor):
        """
        applies convolutions to return the neighbors of every cell.
        :param input_tensor: a tensor of shape (size batch x 2 x input channels x num rows image x num col images)
        :return: a tuple with matrices. each containing one neighbor of every cell in the input in the order
                # 1 | 2
                # 3 | 4
                and with approximate shape (size batch x 2 x input channels x num rows image x num col images)
        :rtype:
        """

        def get_neighbors_from_filter(base_filter):
            """
            given a filter apply to the means and variances separately, stack them and return it
            :param filter:
            :return:
            """
            # filter must have final shape of (output_channels x input_channels x H x W)
            # we keep the number of channels so the number of input and output channels are the same
            num_input_channels = input_tensor.shape[2]
            h = base_filter.shape[0]
            w = base_filter.shape[1]
            filter_mat = torch.zeros((num_input_channels, num_input_channels, h, w))
            # setting the same filter values to all channels
            filter_mat[:, :] = base_filter

            means = input_tensor[:, 0, :, :, :]
            vars = input_tensor[:, 1, :, :, :]
            # send filter_mat to the same device as the means
            filter_mat = filter_mat.to(means.device).double()
            # no padding like in reference implementation
            neigh_means = nn.functional.conv2d(means, filter_mat, stride=self.stride)
            neigh_vars = nn.functional.conv2d(vars, filter_mat, stride=self.stride)
            return torch.stack([neigh_means, neigh_vars], dim=1)

        # need to compute all 2x neighbours of each cell
        # 1 | 2
        # 3 | 4
        # we use some dirty convolutions to that in a efficient way that keeps the backprop intact
        neigh1 = get_neighbors_from_filter(torch.tensor([[1., 0.], [0., 0.]]))
        neigh2 = get_neighbors_from_filter(torch.tensor([[0., 1.], [0., 0.]]))
        neigh3 = get_neighbors_from_filter(torch.tensor([[0., 0.], [1., 0.]]))
        neigh4 = get_neighbors_from_filter(torch.tensor([[0., 0.], [0., 1.]]))
        return neigh1, neigh2, neigh3, neigh4

    def _calculate_max_of_gaussians(self, input_feature_map: torch.tensor) -> torch.tensor:
        """

        :param input_feature_map: a tensor (size batch x 2 x input channels x num rows image x num col images)
        :return: a (size batch x 2 x input channels x num rows image x num col images) tensor
        :rtype:
        """
        neigh1, neigh2, neigh3, neigh4 = self._gen_neighbours(input_feature_map)

        horizontal_max_1_mean, horizontal_max_1_var = self.__max_gaussians_1d(neigh1[:, 0, :, :], neigh1[:, 1, :, :],
                                                                            neigh2[:, 0, :, :], neigh2[:, 1, :, :])
        horizontal_max_2_mean, horizontal_max_2_var = self.__max_gaussians_1d(neigh3[:, 0, :, :], neigh3[:, 1, :, :],
                                                                            neigh4[:, 0, :, :], neigh4[:, 1, :, :])

        mean, var = self.__max_gaussians_1d(horizontal_max_1_mean, horizontal_max_1_var, horizontal_max_2_mean,
                                            horizontal_max_2_var)

        return torch.stack([mean, var], dim=1)

    def __max_gaussians_1d(self, means1: torch.Tensor, vars1: torch.tensor, means2: torch.tensor, vars2: torch.Tensor):
        """
        Computes max of gaussians over a single dimension. Each element in the parameteres corresponds on of the
        parameters of one of the gaussians
        :param means1: a vector of gaussian means
        :param vars1: a vector of gaussian variances
        :param means2:
        :param vars2:
        :return: tuple (means, vars) with means and variances, respectively in tensors
        """
        alpha = torch.sqrt(vars1 + vars2 + self._epsilon)
        beta = (means1 - means2) / alpha
        n = Normal(0, 1)
        cdf_beta = n.cdf(beta)
        cdf_neg_beta = n.cdf(-beta)
        pdf_beta = torch.exp(n.log_prob(beta))

        mean_max = means1 * cdf_beta + means2 * cdf_neg_beta + alpha * pdf_beta

        # the way the variance is calculated here may cause it to become variance. Epsilon is added to try to avoid that
        var_max = (vars1 + torch.pow(means1, 2)) * cdf_beta + (vars2 + torch.pow(means2, 2)) * cdf_neg_beta + \
                  (means1 + means2) * alpha * pdf_beta - torch.pow(mean_max, 2) + self._epsilon
        if torch.any(var_max < 0):
            raise ValueError("Pooling layer: variance is negative. Epsilon should be increased")

        return mean_max, var_max
        

if __name__ == "__main__":
    num_channels = 64
    height = 15
    width = 7
    batch = 100
    input_tensor = torch.rand(100, 2, num_channels, height, width)

    ml = DistributionMaxPool(2)
    r = ml.forward(input_tensor)
    print(r.shape)
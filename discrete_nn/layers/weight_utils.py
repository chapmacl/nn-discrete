"""
This modules implements methods for generating weight distributions from real value weights
"""

from typing import List

import torch
from torch import Tensor


def discretize_weights_shayer(real_weights: Tensor, discrete_weight_values):
    """
    Discretizes the weights from a matrix with real weights based on the Shayer method (pg. 9 of paper). Called by
    discretize_weights_probabilistic.
    :param real_weights: a (output_features x input_features) matrix with real weights
    :param discrete_weight_values: a sorted list of discrete values a weight may take
    :return: a (num_discrete_values x output_features x input_features) with the probabilities
    """

    # the minimum probability for a value
    q_min = 0.05
    # number of discrete values
    number_values = len(discrete_weight_values)
    # maximum probability
    q_max = 1 - (number_values - 1) * q_min
    delta_q = q_max - q_min

    # list of tensors with the probabilities of each of the discrete values
    probability_tensors = []
    for inx, discrete_value in enumerate(discrete_weight_values):
        p_tensor = torch.ones_like(real_weights)
        # add default probability
        p_tensor *= q_min
        if inx == 0:
            # this is the first discretization level. need to account for real values smaller than it
            p_tensor[real_weights <= discrete_value] = q_max
            # weight values greater than the discretized value but smaller than next discretization value
            # the value of the next discretization level

        elif inx == number_values - 1:
            # last discretization level
            p_tensor[real_weights > discrete_value] = q_max

        if inx < number_values - 1:
            # apply to values that are not the last
            next_disc_value = discrete_weight_values[inx + 1]
            # calculating a mask of the weights that are between the current discretization value and the next
            greater_than_current = real_weights > discrete_value
            smaller_than_next_level = real_weights <= next_disc_value
            # update the value for which both conditions are true
            mask = torch.stack((greater_than_current, smaller_than_next_level)).all(dim=0)
            p_tensor[mask] = \
                q_min + delta_q * (next_disc_value - real_weights[mask]) / \
                (next_disc_value - discrete_value)

        if inx > 0:
            # do not apply to the first discretization level
            prev_disc_value = discrete_weight_values[inx - 1]
            smaller_than_current = real_weights <= discrete_value
            greater_than_prev = real_weights > prev_disc_value
            mask = torch.stack((smaller_than_current, greater_than_prev)).all(dim=0)
            # weight values smaller than the discretized value
            p_tensor[mask] = \
                q_min + delta_q * (real_weights[mask] - prev_disc_value) / \
                (discrete_value - prev_disc_value)
        probability_tensors.append(p_tensor)
    return torch.stack(probability_tensors)


def discretize_weights_probabilistic(real_weights: Tensor, discrete_weight_values: List):
    """
    Initializes a linear layer with discrete weights. Redistributes weights and calls discretize_weights_shayer
    :param real_weights: a (output_features x input_features) matrix with real weights
    :param discrete_weight_values: a sorted list of discrete values a weight may take
    :returns: weight distributions (as a mean and standard deviations)
    (num_discrete_values x output_features x input_features)

    """

    def empirical_cdf(weight_row):
        """
        calculates a cumulative density function over the values of real weights on the weight matrix and normalizes
        the weights over the unit interval
        :param: weight_row: a 1D row of weights (num weights)

        :returns: a 1D row of weights with their respective cdf values
        """
        num_weights = weight_row.shape[0]
        sort_val, sort_indices = weight_row.sort(dim=0)  # indices start with zero so we have to shift by 1
        # sort_indices has the indices of the values in the sorted order. The first index is the index to the
        # smallest element

        # to compute the cdf it would be useful, for every value, to have its position in the list. we must invert
        # sort indices

        position_per_value = torch.zeros_like(weight_row, dtype=torch.float64)
        position_per_value[sort_indices] = torch.tensor(list(range(num_weights))).double().to("cuda:0") + 1.0
        cdf = position_per_value / num_weights

        return cdf

    delta_w = discrete_weight_values[1] - discrete_weight_values[0]

    # compute cdf separately for positive and negative weights
    weight_cdf = torch.zeros_like(real_weights, dtype=torch.float64)
    # normalizing according to paper [w_1 - \delta_w/2, w_D - \delta_w/2]
    weight_cdf[real_weights <= 0] = empirical_cdf(real_weights[real_weights <= 0]) * \
                                    (-discrete_weight_values[0] + delta_w / 2) - (
                                            -discrete_weight_values[0] + delta_w / 2)
    weight_cdf[real_weights > 0] = empirical_cdf(real_weights[real_weights > 0]) * \
                                   (discrete_weight_values[-1] + delta_w / 2)
    # need to shift back to a matrix
    # weight_cdf = weight_cdf.reshape(real_weights.shape)

    # we use shayers discretization method with the shifted weights
    shayers_discretized = discretize_weights_shayer(weight_cdf)

    # we need to have log-probabilities
    shayers_discretized = torch.log(shayers_discretized)

    return shayers_discretized

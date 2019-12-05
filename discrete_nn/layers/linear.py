"""
Implements different linear layer classes by extending from torch.nn.Module
"""
from torch import Tensor
import torch
from torch import nn


class DiscreteLinear(nn.Module):
    def __init__(self, input_features, output_features, init_weights, discretized_values):
        """
        Initializes a linear layer with discrete weights
        :param input_features: the input dimension
        :param output_features: the output dimension (# of neurons in layer)
        :param init_weights: a tensor (output_features x input_features) with pre trained real weights
        :param discretized_values: a list of the discretized values
        """
        super().__init__()
        self.in_feat = input_features
        self.ou_feat = output_features
        self.discrete_values = discretized_values

    def discretize_weights_shayer(self, real_weights: Tensor):
        """
        todo finish this
        Discretizes the weights from a matrix with real weights based on the Shayer method (pg. 9 of paper)

        :param real_weights: a (output_features x input_features) matrix with real weights
        :return: a (num_discrete_values x output_features x input_features) with the probabilities
        """

        # the minimum probability for a value
        q_min = 0.05
        # number of discrete values
        number_values = len(self.discrete_values)
        # maximum probability
        q_max = 1 - (number_values-1)*q_min
        delta_q = q_max - q_min

        # list of tensors with the probabilities of each of the discrete values
        probability_tensors = []

        for inx, discrete_value in enumerate(self.discrete_values):
            p_tensor = torch.zeros_like(real_weights)
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
                next_disc_value = self.discrete_values[inx + 1]
                # calculating a mask of the weights that are between the current discretization value and the next
                greater_than_current = real_weights > discrete_value
                smaller_than_next_level = real_weights <= next_disc_value
                # update the value for which both conditions are true
                mask = torch.stack((greater_than_current, smaller_than_next_level)).all(dim=0)
                p_tensor[mask] = \
                    q_min + delta_q * (- real_weights[mask]) / \
                    (next_disc_value - discrete_value)

            if inx > 0:
                # do not apply to the first discretization level
                prev_disc_value = self.discrete_values[inx - 1]
                smaller_than_current = real_weights <= discrete_value
                greater_than_prev = real_weights > prev_disc_value
                mask = torch.stack((smaller_than_current, greater_than_prev)).all(dim=0)
                # weight values smaller than the discretized value
                p_tensor[mask] = \
                    q_min + delta_q * (real_weights[mask] - prev_disc_value) / \
                    (discrete_value - prev_disc_value)

    def discretize_weights_probabilistic(self, real_weights: Tensor):
        """
        Initializes a linear layer with discrete weights
        :param input_features: the input dimension
        :param output_features: the output dimension (# of neurons in layer)
        :param init_weights: a tensor (output_features x input_features) with pre trained real weights
        :param discretized_values: a list of the discretized values
        """
        pass

class TernaryLinear(DiscreteLinear):
    def __init__(self, input_features, output_features, init_weights):
        """
        Discretizes the weights from a matrix with real weights based on the "probabilistic" method proposed in
        the paper (pg. 9, last paragraph)

        :param real_weights: a (output_features x input_features) matrix with real weights
        :return: a (num_discrete_values x output_features x input_features) with the probabilities
        """
        super().__init__(input_features, output_features, init_weights, [-1, 0, 1])




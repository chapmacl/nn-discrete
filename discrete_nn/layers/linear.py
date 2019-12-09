"""
Implements different linear layer classes by extending from torch.nn.Module
"""
from torch import Tensor
import torch
from torch import nn


class DiscreteLinear(nn.Module):
    def __init__(self, input_features, output_features, discretized_values):
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
        self.discrete_values = sorted(discretized_values)

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
        print(f"q_max{q_max}")
        delta_q = q_max - q_min
        print(f"delta_q{delta_q}")
        # list of tensors with the probabilities of each of the discrete values
        probability_tensors = []
        for inx, discrete_value in enumerate(self.discrete_values):
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
                next_disc_value = self.discrete_values[inx + 1]
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
                prev_disc_value = self.discrete_values[inx - 1]
                smaller_than_current = real_weights <= discrete_value
                greater_than_prev = real_weights > prev_disc_value
                mask = torch.stack((smaller_than_current, greater_than_prev)).all(dim=0)
                # weight values smaller than the discretized value
                p_tensor[mask] = \
                    q_min + delta_q * (real_weights[mask] - prev_disc_value) / \
                    (discrete_value - prev_disc_value)
            probability_tensors.append(p_tensor)
        return torch.stack(probability_tensors)

    def discretize_weights_probabilistic(self, real_weights: Tensor):
        """
        Initializes a linear layer with discrete weights. Redistributes weights and calls discretize_weights_shayer
        :param real_weights: a (output_features x input_features) matrix with real weights

        """

        def empirical_cdf(weight_row):
            """
            calculates a cumulative density function over the values of real weights on the weight matrix and normalizes
            the weights over the unit interval
            :param: weight_row: a 1D row of weights (num weights)

            :returns: a 1D row of weights with their respective cdf values
            """

            num_weights = weight_row.shape[0]
            # flattening weight matrix
            # flat_weight_row = weight_matrix.reshape(1, num_weights)
            # repeat the flat vector num_weight times into consecutive rows
            repeated_weight_matrix = weight_row.repeat(num_weights, 1)
            
            # we calculate each weights cdf by subtracting its value from one of the rows in repeated_weight_matrix
            flat_weight_vector = weight_row.reshape(-1, 1)
            diff_matrix = repeated_weight_matrix - flat_weight_vector
            # count all times a weight from the vector was greater than one in the row by checking the amount of
            # <= 0 values on the diff matrix
            count_row = (diff_matrix <= 0).sum(dim=1)
            # in this row, for each location,  we have the number of time the corresponding weight was greater or
            # equal to another

            # we normalize to get the CDF
            return count_row.double() / num_weights

        # todo handle weights of different signs separetely?
        """
        discretized_weights = torch.zeros_like(real_weights)
        # we apply this separately to positive and negative weights to preserve the sign
        negative_discrete_weights = [weight for weight in self.discrete_values if weight < 0]
        positive_discrete_weights = [weight for weight in self.discrete_values if weight >= 0]
    
        # normalize positive weights
        # scaling to unit interval
        discretized_weights[real_weights >= 0] = empirical_cdf(real_weights[real_weights >= 0])
        
        delta_w_positive = positive_discrete_weights[-1] - 
        """
        weight_cdf = empirical_cdf(real_weights.reshape(-1))
        # need to shift back to a matrix
        weight_cdf = weight_cdf.reshape(real_weights.shape)
        # shift weights to interval [w_1 - \delta_w/2 , w_D - \delta_w/2]
        delta_w = self.discrete_values[-1] - self.discrete_values[0]
        # paper DOES NOT DEFINE WHAT THIS IS SUPPOSED TO BE

        lower_bound_interval = self.discrete_values[0] - delta_w/2
        upper_bound_interval = self.discrete_values[-1] + delta_w/2
        scaled_weights = weight_cdf * (upper_bound_interval - lower_bound_interval)
        scaled_weights += lower_bound_interval

        # we use shayers discretization method with the shifted weights
        return self.discretize_weights_shayer(scaled_weights)


class TernaryLinear(DiscreteLinear):
    def __init__(self, input_features, output_features, init_weights):
        """
        Discretizes the weights from a matrix with real weights based on the "probabilistic" method proposed in
        the paper (pg. 9, last paragraph)

        :param real_weights: a (output_features x input_features) matrix with real weights
        :return: a (num_discrete_values x output_features x input_features) with the probabilities
        """
        super().__init__(input_features, output_features, init_weights, [-1, 0, 1])


if __name__ == "__main__":
    test_weight_matrix = torch.tensor([[2.1, 0.3, -0.5, -2], [4., 0., 2., -1.]])
    layer = DiscreteLinear(4, 2, [-1, 0, 1])
    layer.discretize_weights_probabilistic(test_weight_matrix)
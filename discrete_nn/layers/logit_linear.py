"""
Implements different linear layer classes by extending from torch.nn.Module
"""
from discrete_nn.layers.types import ValueTypes

from torch import nn
from torch import Tensor
import torch


class LogitLinear(nn.Module):
    def __init__(self, input_features, input_feature_type: ValueTypes, output_features, initialization_weights,
                 initialization_bias, discretized_values, normalize_activations: bool = False):
        """
        Initializes a linear layer with discrete weights
        :param input_features: the input dimension
        :param input_feature_type: the type of the input (e.g. real, gaussian distribution)
        :param output_features: the output dimension (# of neurons in layer)
        :param initialization_weights: a tensor (output_features x input_features) with pre trained real weights to b/e
        used for initializing the discrete ones
        :param initialization_bias: a tensor (output_features x 1) with pre trained real weights for bias to be
        used for initializing the discrete ones
        :param discretized_values: a list of the discretized values
        :param normalize_activations: if true normalize the activations by dividing by the number of the layer's inputs
        (including bias)
        """
        super().__init__()
        self.in_feat = input_features
        self.in_feat_type = input_feature_type
        self.ou_feat = output_features
        self.discrete_values = sorted(discretized_values)
        # todo what about bias?
        # these are the tunable parameters
        self.W_logits = torch.nn.Parameter(self.discretize_weights_probabilistic(initialization_weights),
                                           requires_grad=True)
        self.b_logits = torch.nn.Parameter(self.discretize_weights_probabilistic(initialization_bias),
                                           requires_grad=True)
        self.normalize_activations = normalize_activations

    def sample_discrete_weights(self):
        """ sample a discrete weight from the weights of the layer based on the weight distributions"""
        probabilities_w = self.generate_weight_probabilities(self.W_logits)
        probabilities_b = self.generate_weight_probabilities(self.b_logits)



    def discretize_weights_shayer(self, real_weights: Tensor):
        """
        Discretizes the weights from a matrix with real weights based on the Shayer method (pg. 9 of paper). Called by
        discretize_weights_probabilistic.
        :param real_weights: a (output_features x input_features) matrix with real weights
        :return: a (num_discrete_values x output_features x input_features) with the probabilities
        """

        # the minimum probability for a value
        q_min = 0.05
        # number of discrete values
        number_values = len(self.discrete_values)
        # maximum probability
        q_max = 1 - (number_values - 1) * q_min
        delta_q = q_max - q_min

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
            position_per_value[sort_indices] = torch.tensor(list(range(num_weights))).double() + 1.0
            cdf = position_per_value / num_weights
            # cumulative_sum = sort_val.abs().cumsum(dim=0)
            # cdf_val_sorted = cumulative_sum / sort_val.sum().abs()

            # cdf = torch.zeros_like(weight_row, dtype=torch.float64)
            # cdf[sort_indices] = cdf_val_sorted.double()

            return cdf

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
        delta_w = self.discrete_values[1] - self.discrete_values[0]

        # compute cdf separately for positive and negative weights
        weight_cdf = torch.zeros_like(real_weights, dtype=torch.float64)
        # normalizing according to paper [w_1 - \delta_w/2, w_D - \delta_w/2]
        weight_cdf[real_weights <= 0] = empirical_cdf(real_weights[real_weights <= 0]) * \
                                       (-self.discrete_values[0] + delta_w / 2) - (
                                               -self.discrete_values[0] + delta_w / 2)
        weight_cdf[real_weights > 0] = empirical_cdf(real_weights[real_weights > 0]) * \
                                       (self.discrete_values[-1] + delta_w / 2)
        # need to shift back to a matrix
        # weight_cdf = weight_cdf.reshape(real_weights.shape)

        # we use shayers discretization method with the shifted weights
        shayers_discretized = self.discretize_weights_shayer(weight_cdf)

        # we need to have log-probabilities
        shayers_discretized = torch.log(shayers_discretized)

        return shayers_discretized

    def generate_weight_probabilities(self, logit_weights):
        """
        calculates the probabilities of all discrete weights for the logits provided

        :param logit_weights: a tensor with dimensions (discretization levels x output features x input_features)
        with the discrete distribution as logits
        :return:  a tensor with dimensions (discretization levels x output features x input_features)
        with the discrete distribution as probabilities
        """
        weight_probabilities = torch.exp(logit_weights)
        weight_probabilities = weight_probabilities / weight_probabilities.sum(dim=0)
        return weight_probabilities

    def get_gaussian_dist_parameters(self, logit_weights):
        """
        Fits a gaussian distribution to the logits in logit_weights.
        :param logit_weights: a tensor with dimensions (discretization levels x output features x input_features)
        with the discrete distribution as logits
        :return: a tuple with the means of the gaussian distributions as a (output features x input_features) tensor
        and a the standard deviations in a tensor of the same format
        """
        weight_probabilities = self.generate_weight_probabilities(logit_weights)
        discrete_val_tensor = torch.zeros_like(logit_weights)
        for inx, discrete_weight in enumerate(self.discrete_values):
            discrete_val_tensor[inx, :, :] = discrete_weight
        discrete_val_tensor.requires_grad = True
        weight_mean = discrete_val_tensor * weight_probabilities
        weight_mean = weight_mean.sum(dim=0)
        # for inx, discrete_weight in enumerate(self.discrete_values):
        # weight prob * weight value
        #    weight_mean += weight_probabilities[inx, :, :] * discrete_weight

        weight_var = weight_probabilities * torch.pow(discrete_val_tensor - weight_mean, 2)
        weight_var = weight_var.sum(dim=0)
        # for inx, discrete_weight in enumerate(self.discrete_values):
        #    # weight prob * (weight value - mean)^2
        #    weight_var += weight_probabilities[inx, :, :] * torch.pow(discrete_weight - weight_mean, 2)

        return weight_mean, weight_var

    def forward(self, x: torch.Tensor):
        """
        Applies an input to the network. Depending on the type of the input value a different input value is expected
        :param x: (n x in_feat) if the values are real, (n x 2 x in_feat) where the second dimension contains, in order,
        the mean and variance,  n is the number of samples and, in_feat is the number of input features.
        :return: a tensor with dimensions (n x out_feat) if the values are real, (n x 2 x out_feat) where the second
        dimension contains, in order,
        the mean and variance,  n is the number of samples and, out_feat is the number of out features.
        """
        output_mean = None
        output_var = None

        w_mean, w_var = self.get_gaussian_dist_parameters(self.W_logits)
        b_mean, b_var = self.get_gaussian_dist_parameters(self.b_logits)
        # print(f"maximum abs logit weight {self.W_logits.abs().max()}")
        """
        print(f"maximum logit weight {self.W_logits.max()}")
        print(f"minimum logit weight {self.W_logits.min()}")

        print(f"maximum mean weight {w_mean.max()}")
        print(f"mininmum mean weight {w_mean.min()}")
        """
        if self.in_feat_type == ValueTypes.REAL:
            # transpose the input matrix to (in_feat x n)
            input_values = x.transpose(0, 1).double()
            # outputs are a tensor (out_feat x n)
            output_mean = torch.mm(w_mean, input_values)
            output_var = torch.mm(w_var, torch.pow(input_values, 2))

        elif self.in_feat_type == ValueTypes.GAUSSIAN:
            x_mean = x[:, 0, :]
            x_var = x[:, 1, :]
            # transpose the input matrices to (in_feat x n)
            x_mean = x_mean.transpose(1, 0)
            x_var = x_var.transpose(1, 0)
            output_mean = torch.mm(w_mean, x_mean)
            output_var = torch.mm(torch.pow(x_mean, 2), w_var) + torch.mm(x_var, torch.pow(w_mean, 2)) + \
                         torch.mm(x_var, w_var)
        else:
            raise ValueError(f"The type {self.in_feat_type} given for the input type of layer is invalid")
        # need to return the matrix back to the original axis layout by transposing again
        output_mean = output_mean.transpose(0, 1)
        output_var = output_var.transpose(0, 1)

        output_mean += b_mean[:, 0]  # broadcasting to all samples
        output_var += b_var[:, 0]

        if self.normalize_activations:
            output_mean /= (self.in_feat + 1) ** 0.5
            output_var /= (self.in_feat + 1)

        return torch.stack([output_mean, output_var], dim=1)


class TernaryLinear(LogitLinear):
    def __init__(self, input_features: int, input_feature_type: ValueTypes, output_features: int, real_init_weights,
                 real_init_bias, normalize_activations: bool = False):
        """
        Discretizes the weights from a matrix with real weights based on the "probabilistic" method proposed in
        the paper (pg. 9, last paragraph)
        :param input_features: the number of inputs
        :param input_feature_type: the type of the input (e.g. real, gaussian distribution)
        :param output_features: the number of outputs
        :param real_init_weights: the real initialization weights
        :param real_init_bias: the real initialization bias weights
        :param normalize_activations: if true normalize the activations by dividing by the number of the layer's inputs
        (including bias)
        :return: a (num_discrete_values x output_features x input_features) with the probabilities

        """
        super().__init__(input_features, input_feature_type, output_features, real_init_weights, real_init_bias,
                         [-1, 0, 1], normalize_activations)


if __name__ == "__main__":
    test_weight_matrix = torch.tensor([[2.1, 30, -0.5, -2], [4., 0., 2., -1.], [2.1, 0.3, -0.5, -2]])
    test_bias_matrix = torch.tensor([[2.1, 0.3, -0.5]])
    test_samples = torch.rand(10, 4).double()  # 10 samples, 4 in dimensions
    layer = LogitLinear(4, ValueTypes.REAL, 3, test_weight_matrix, test_bias_matrix, [-2, -1, 0, 1, 2])
    x = layer.forward(test_samples)
    print(x.shape)

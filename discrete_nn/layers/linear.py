"""
Implements different linear layer classes by extending from torch.nn.Module
"""
from discrete_nn.layers.types import ValueTypes

from torch import nn
from torch import Tensor
import torch


class DiscreteLinear(nn.Module):
    def __init__(self, input_features, input_feature_type: ValueTypes, output_features, initialization_weights,
                 initialization_bias, discretized_values):
        """
        Initializes a linear layer with discrete weights
        :param input_features: the input dimension
        :param input_feature_type: the type of the input (e.g. real, gaussian distribution)
        :param output_features: the output dimension (# of neurons in layer)
        :param initialization_weights: a tensor (output_features x input_features) with pre trained real weights to be
        used for initializing the discrete ones
        :param initialization_bias: a tensor (output_features x 1) with pre trained real weights for bias to be
        used for initializing the discrete ones
        :param discretized_values: a list of the discretized values
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
        :returns: weight distributions (as a mean and standard deviations)

        """

        def empirical_cdf(weight_row):
            """
            calculates a cumulative density function over the values of real weights on the weight matrix and normalizes
            the weights over the unit interval
            :param: weight_row: a 1D row of weights (num weights)

            :returns: a 1D row of weights with their respective cdf values
            """
            num_weights = weight_row.shape[0]

            sort_indices = weight_row.argsort(dim=0) + 1  # indices start with zero so we have to shift by 1

            cdf = sort_indices.double() / num_weights
            return cdf
            """
            
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
            # in this row, for each location,  we have the number of times the corresponding weight was greater or
            # equal to another

            # we normalize to get the CDF
            return count_row.double() / num_weights
            """

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
        shayers_discretized = self.discretize_weights_shayer(scaled_weights)

        # we need to have log-probabilities
        shayers_discretized = torch.log(shayers_discretized)

        return shayers_discretized

    def generate_weight_distribution(self, logit_weights):
        """
        Fits a gaussian distribution to the logits in logit_weights.
        :param logit_weights: a tensor with dimensions (dicretization levels x output features x input_features)
        with the discrete distribution as logits
        :return: a tuple with the means of the gaussian distributions as a (output features x input_features) tensor
        and a the standard deviations in a tensor of the same format
        """

        # create placeholder tensors for mean and stddevs
        weight_probabilities = torch.exp(logit_weights)

        weight_mean = torch.zeros_like(weight_probabilities[0])
        weight_var = torch.zeros_like(weight_probabilities[0])
        for inx, discrete_weight in enumerate(self.discrete_values):
            # weight prob * weight value
            weight_mean += weight_probabilities[inx, :, :] * discrete_weight
        for inx, discrete_weight in enumerate(self.discrete_values):
            # weight prob * (weight value - mean)^2
            weight_var += weight_probabilities[inx, :, :] * torch.pow(discrete_weight-weight_mean, 2)

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

        w_mean, w_var = self.generate_weight_distribution(self.W_logits)
        b_mean, b_var = self.generate_weight_distribution(self.b_logits)

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
            output_var = torch.mm(torch.pow(x_mean, 2), w_var) + torch.mm(x_var, torch.pow(w_mean, 2)) +\
                torch.mm(x_var, w_var)
        else:
            raise ValueError(f"The type {self.in_feat_type} given for the input type of layer is invalid")
        # need to return the matrix back to the original axis layout by transposing again
        output_mean = output_mean.transpose(0, 1)
        output_var = output_var.transpose(0, 1)

        output_mean += b_mean[:, 0]  # broadcasting to all samples
        output_var += b_var[:, 0]

        return torch.stack([output_mean, output_var], dim=1)


class TernaryLinear(DiscreteLinear):
    def __init__(self, input_features, input_feature_type: ValueTypes, output_features, real_init_weights, real_init_bias):
        """
        Discretizes the weights from a matrix with real weights based on the "probabilistic" method proposed in
        the paper (pg. 9, last paragraph)
        :param input_features: the number of inputs
        :param input_feature_type: the type of the input (e.g. real, gaussian distribution)
        :param output_features: the number of outputs
        :param real_init_weights: the real initialization weights
        :return: a (num_discrete_values x output_features x input_features) with the probabilities
        """
        super().__init__(input_features, input_feature_type, output_features, real_init_weights,real_init_bias,
                         [-1, 0, 1])


if __name__ == "__main__":
    test_weight_matrix = torch.tensor([[2.1, 30, -0.5, -2], [4., 0., 2., -1.], [2.1, 0.3, -0.5, -2]])
    test_bias_matrix = torch.tensor([[2.1, 0.3, -0.5]])
    test_samples = torch.rand(10, 4).double()  # 10 samples, 4 in dimensions
    layer = DiscreteLinear(4, ValueTypes.REAL, 3, test_weight_matrix, test_bias_matrix, [-2, -1, 0, 1, 2])
    x = layer.forward(test_samples)
    print(x.shape)

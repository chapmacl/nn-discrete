"""
Implements different linear layer classes by extending from torch.nn.Module
"""
from discrete_nn.layers.type_defs import ValueTypes
from discrete_nn.layers.weight_utils import discretize_weights_probabilistic

from torch import nn
from torch.distributions import Multinomial
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
        :param discretized_values: a list of the discrete weight values (e.g. [-1, 1, 0])
        :param normalize_activations: if true normalize the activations by dividing by the number of the layer's inputs
        (including bias)
        """
        super().__init__()
        self.in_feat = input_features
        self.in_feat_type = input_feature_type
        self.ou_feat = output_features
        self.discrete_values = sorted(discretized_values)
        self.normalize_activations = normalize_activations
        # these are the tunable parameters
        self.W_logits = torch.nn.Parameter(discretize_weights_probabilistic(initialization_weights,
                                                                            self.discrete_values),
                                           requires_grad=True)
        self.b_logits = torch.nn.Parameter(discretize_weights_probabilistic(initialization_bias, self.discrete_values),
                                           requires_grad=True)

    def generate_discrete_network(self, method: str = "sample"):
        """ generates discrete weights from the weights of the layer based on the weight distributions

        :param method: the method to use to generate the discrete weights. Either argmax or sample

        :returs: tuple (sampled_w, sampled_b) where sampled_w and sampled_b are tensors of the shapes
        (output_features x input_features) and (output_features x 1)
        """
        probabilities_w = self.generate_weight_probabilities(self.W_logits)
        probabilities_b = self.generate_weight_probabilities(self.b_logits)
        assert(probabilities_b.shape != probabilities_w.shape)
        # logit probabilities must be in inner dimension for torch.distribution.Multinomial
        probabilities_w = probabilities_w.transpose(0, 1).transpose(1, 2)
        # stepped transpose bc we need to keep the order of the other dimensions
        probabilities_b = probabilities_b.transpose(0, 1).transpose(1, 2)
        discrete_values_tensor = torch.tensor(self.discrete_values).to(self.device).double()
        discrete_values_tensor.to(self.device)
        if method == "sample":
            m_w = Multinomial(probs=probabilities_w)
            m_b = Multinomial(probs=probabilities_b)
            # this is a output_features x input_features x discretization_levels mask
            sampled_w = m_w.sample()
            sampled_b = m_b.sample()
            # need to make sure these tensors are in the cpu
            

            if torch.all(sampled_b.sum(dim=2) != 1):
                raise ValueError("sampled mask for bias does not sum to 1")
            if torch.all(sampled_w.sum(dim=2) != 1):
                raise ValueError("sampled mask for weights does not sum to 1")

            # need to generate the discrete weights from the masks

            sampled_w = torch.matmul(sampled_w, discrete_values_tensor)
            sampled_b = torch.matmul(sampled_b, discrete_values_tensor)
        elif method == "argmax":
            # returns a (out_feat x in_feat) matrix where the values correspond to the index of the discretized value
            # with the largest probability
            argmax_w = torch.argmax(probabilities_w, dim=2)
            argmax_b = torch.argmax(probabilities_b, dim=2)
            # creating placeholder for discrete weights
            sampled_w = torch.zeros_like(argmax_w).to("cpu")
            sampled_b = torch.zeros_like(argmax_b).to("cpu")
            sampled_w[:] = discrete_values_tensor[argmax_w[:]]
            sampled_b[:] = discrete_values_tensor[argmax_b[:]]

        else:
            raise ValueError(f"Invalid method {method} for layer discretization")

        # sanity checks
        if sampled_w.shape != (probabilities_w.shape[0], probabilities_w.shape[1]):
            raise ValueError("sampled probability mask for weights does not match expected shape")
        if sampled_b.shape != (probabilities_b.shape[0], probabilities_b.shape[1]):
            raise ValueError("sampled probability mask for bias does not match expected shape")

        return sampled_w, sampled_b

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

        weight_var = weight_probabilities * torch.pow(discrete_val_tensor - weight_mean, 2)
        weight_var = weight_var.sum(dim=0)
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
        w_mean, w_var = self.get_gaussian_dist_parameters(self.W_logits)
        b_mean, b_var = self.get_gaussian_dist_parameters(self.b_logits)
        # print(f"maximum abs logit weight {self.W_logits.abs().max()}")
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
        Discretizes the weights from a matr    if device == "cuda:0":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)ix with real weights based on the "probabilistic" method proposed in
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
    x_out = layer.forward(test_samples)
    print(x_out.shape)
    # discretizing weights
    print("sampled discrete network", layer.generate_discrete_network("sample"))
    print("argmax discrete network", layer.generate_discrete_network("argmax"))

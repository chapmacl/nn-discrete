"""
Implements different linear layer classes by extending from torch.nn.Module
"""
from typing import Optional

from discrete_nn.layers.type_defs import ValueTypes, DiscreteWeights, WeightTypes
from discrete_nn.layers.weight_utils import discretize_weights_probabilistic, generate_weight_probabilities, get_gaussian_dist_parameters

from torch import nn
from torch.distributions import Multinomial
import torch


class LogitLinear(nn.Module):
    def __init__(self, input_features, input_feature_type: ValueTypes, output_features, initialization_weights,
                 initialization_bias: Optional[torch.Tensor], bias_weight_type: Optional[WeightTypes],
                 discretized_values: DiscreteWeights,
                 normalize_activations: bool = False):
        """
        Initializes a linear layer with discrete weights
        :param input_features: the input dimension
        :param input_feature_type: the type of the input (e.g. real, gaussian distribution)
        :param output_features: the output dimension (# of neurons in layer)
        :param initialization_weights: a tensor (output_features x input_features) with pre trained real weights to b/e
        used for initializing the discrete ones
        :param initialization_bias: a tensor (output_features x 1) with pre trained real weights for bias to be
        used for initializing the discrete ones. If None, disables bias
        :param bias_weight_type: the weight type of the bias: Either real or logit. None if bias is not enabled
        :param discretized_values: a list of the discrete weight values (e.g. [-1, 1, 0])
        :param normalize_activations: if true normalize the activations by dividing by the number of the layer's inputs
        (including bias)
        """
        super().__init__()
        self.in_feat = input_features
        self.in_feat_type = input_feature_type
        self.ou_feat = output_features
        self.discrete_values = sorted(discretized_values.value)
        self.bias_w_type = bias_weight_type
        self.normalize_activations = normalize_activations
        # these are the tunable parameters
        self.W_logits = torch.nn.Parameter(discretize_weights_probabilistic(initialization_weights,
                                                                            self.discrete_values),
                                           requires_grad=True)
        if initialization_bias is not None:
            if bias_weight_type is None:
                raise ValueError("Initialization values for bias were provided but not its type")
            if bias_weight_type == WeightTypes.LOGIT:
                self.b_logits = torch.nn.Parameter(discretize_weights_probabilistic(initialization_bias, self.discrete_values),
                                                   requires_grad=True)
            elif bias_weight_type == WeightTypes.REAL:
                self.b_logits = torch.nn.Parameter(initialization_bias, requires_grad=True)
            else:
                raise ValueError(f"unsupported weight type {bias_weight_type} for bias")
        else:
            self.b_logits = None

    def generate_discrete_network(self, method: str = "sample"):
        """ generates discrete weights from the weights of the layer based on the weight distributions

        :param method: the method to use to generate the discrete weights. Either argmax or sample

        :returs: tuple (sampled_w, sampled_b) where sampled_w and sampled_b are tensors of the shapes
        (output_features x input_features) and (output_features x 1). sampled_b is None if bias is disable
        """
        probabilities_w = generate_weight_probabilities(self.W_logits).cpu()

        if self.b_logits is not None and self.bias_w_type == WeightTypes.LOGIT:
            probabilities_b = generate_weight_probabilities(self.b_logits).cpu()
            assert(probabilities_b.shape != probabilities_w.shape)
            probabilities_b = probabilities_b.transpose(0, 1).transpose(1, 2)
        else:
            probabilities_b = None

        # logit probabilities must be in inner dimension for torch.distribution.Multinomial
        # stepped transpose bc we need to keep the order of the other dimensions
        probabilities_w = probabilities_w.transpose(0, 1).transpose(1, 2)

        discrete_values_tensor = torch.tensor(self.discrete_values).double()

        if method == "sample":
            m_w = Multinomial(probs=probabilities_w)
            if probabilities_b is not None:
                m_b = Multinomial(probs=probabilities_b)
                sampled_b = m_b.sample()
                if torch.all(sampled_b.sum(dim=2) != 1):
                    raise ValueError("sampled mask for bias does not sum to 1")
                sampled_b = torch.matmul(sampled_b, discrete_values_tensor)
            else:
                sampled_b = None

            # this is a output_features x input_features x discretization_levels mask
            sampled_w = m_w.sample()
            # need to make sure these tensors are in the cpu
            if torch.all(sampled_w.sum(dim=2) != 1):
                raise ValueError("sampled mask for weights does not sum to 1")

            # need to generate the discrete weights from the masks
            sampled_w = torch.matmul(sampled_w, discrete_values_tensor)

        elif method == "argmax":
            # returns a (out_feat x in_feat) matrix where the values correspond to the index of the discretized value
            # with the largest probability
            argmax_w = torch.argmax(probabilities_w, dim=2)
            if probabilities_b is not None:
                argmax_b = torch.argmax(probabilities_b, dim=2)
                sampled_b = torch.zeros_like(argmax_b).to("cpu")
                sampled_b[:] = discrete_values_tensor[argmax_b[:]]
            else:
                sampled_b = None
            # creating placeholder for discrete weights
            sampled_w = torch.zeros_like(argmax_w).to("cpu")
            sampled_w[:] = discrete_values_tensor[argmax_w[:]]

        else:
            raise ValueError(f"Invalid method {method} for layer discretization")

        # sanity checks
        if sampled_w.shape != (probabilities_w.shape[0], probabilities_w.shape[1]):
            raise ValueError("sampled probability mask for weights does not match expected shape")
        if sampled_b is not None and sampled_b.shape != (probabilities_b.shape[0], probabilities_b.shape[1]):
            raise ValueError("sampled probability mask for bias does not match expected shape")

        if self.bias_w_type == WeightTypes.LOGIT or self.b_logits is None:
            return sampled_w, sampled_b
        else:
            return sampled_w, self.b_logits.clone().detach()

    def forward(self, x: torch.Tensor):
        """
        Applies an input to the network. Depending on the type of the input value a different input value is expected
        :param x: (n x in_feat) if the values are real, (n x 2 x in_feat) where the second dimension contains, in order,
        the mean and variance,  n is the number of samples and, in_feat is the number of input features.
        :return: a tensor with dimensions (n x out_feat) if the values are real, (n x 2 x out_feat) where the second
        dimension contains, in order,
        the mean and variance,  n is the number of samples and, out_feat is the number of out features.
        """
        w_mean, w_var = get_gaussian_dist_parameters(self.W_logits, self.discrete_values)

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

        if self.b_logits is not None:
            if self.bias_w_type == WeightTypes.LOGIT:
                b_mean, b_var = get_gaussian_dist_parameters(self.b_logits, self.discrete_values)
                output_mean += b_mean[:, 0]  # broadcasting to all samples
                output_var += b_var[:, 0]
            else:
                output_mean += self.b_logits[:, 0]

        if self.normalize_activations:
            output_mean /= (self.in_feat + 1) ** 0.5
            output_var /= (self.in_feat + 1)

        return torch.stack([output_mean, output_var], dim=1)


if __name__ == "__main__":
    test_weight_matrix = torch.tensor([[2.1, 30, -0.5, -2], [4., 0., 2., -1.], [2.1, 0.3, -0.5, -2]])
    test_bias_matrix = torch.tensor([[2.1, 0.3, -0.5]])
    test_samples = torch.rand(10, 4).double()  # 10 samples, 4 in dimensions
    layer = LogitLinear(4, ValueTypes.REAL, 3, test_weight_matrix, test_bias_matrix, DiscreteWeights.TERNARY)
    x_out = layer.forward(test_samples)
    print(x_out.shape)
    # discretizing weights
    print("sampled discrete network", layer.generate_discrete_network("sample"))
    print("argmax discrete network", layer.generate_discrete_network("argmax"))

from torch.nn import Module
import torch


class Linear(Module):
    def __init__(self, number_inputs, number_outputs, normalize_activations=False):
        super().__init__()
        self.normalize_activations = normalize_activations
        self.number_inputs = number_inputs
        self.weight = torch.nn.Parameter(torch.rand(number_outputs, number_inputs), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.rand(number_outputs), requires_grad=True)

    def forward(self, x):
        # transpose the input matrix to (in_feat x n)
        input_values = x.transpose(0, 1)
        out = torch.mm(self.weight, input_values)
        # need to return the matrix back to the original axis layout by transposing again
        out = out.transpose(0, 1)

        out += self.bias

        if self.normalize_activations:
            out = out / (self.number_inputs + 1)**0.5
        return out

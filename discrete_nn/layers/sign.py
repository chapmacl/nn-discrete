import torch
from torch import nn

from discrete_nn.layers.type_defs import InputFormat


class DistributionSign(nn.Module):
    def __init__(self, input_format: InputFormat):
        super().__init__()
        self._input_format = input_format

    def forward(self, x):
        if self._input_format == InputFormat.FEATURE_MAP:
            means = x[:, 0, :, :, :]
            vars = x[:, 1, :, :, :]

        elif self._input_format == InputFormat.FLAT_ARRAY:
            means = x[:, 0, :]
            vars = x[:, 1, :]
        else:
            raise ValueError(f"input format :{self._input_format} is not valid")
        mean_out, var_out = self._sign(means, vars)
        return torch.stack([mean_out, var_out], dim=1)

    def _sign(self, mean: torch.Tensor, variance: torch.Tensor):
        eps = 1e-8
        mean_out: torch.Tensor = torch.erf(mean / torch.sqrt(2. * variance + eps))
        var_out = 1. - torch.pow(mean_out, 2) + 1e-6
        return mean_out, var_out


if __name__ == "__main__":
    num_channels = 4
    d = DistributionSign(InputFormat.FEATURE_MAP)
    h = 10
    w = 11
    batch_s = 100
    input_tensor = torch.rand(batch_s, 2, num_channels, h, w)
    d.forward(input_tensor)
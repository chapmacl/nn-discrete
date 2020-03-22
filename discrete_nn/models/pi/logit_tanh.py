import torch
from discrete_nn.layers.type_defs import ValueTypes, DiscreteWeights
from discrete_nn.layers.logit_linear import LogitLinear
from discrete_nn.layers.local_reparametrization import LocalReparametrization
from discrete_nn.models.pi.real import PiReal
from discrete_nn.models.base_model import LogitModel


class PiLogitTanh(LogitModel):
    """
    Logit-based weighted (non convolutionary) network
    """

    def __init__(self, real_model_params, discrete_weights: DiscreteWeights):
        """

        :param real_model_params: a dictionary containing the real weights of the pretrained model
        """
        super().__init__()
        self.discrete_weights: DiscreteWeights = discrete_weights

        s1_l1_dropout = torch.nn.Dropout(p=0.1)
        s2_l1_linear = LogitLinear(784, ValueTypes.REAL, 1200, real_model_params["L1_Linear_W"],
                                   real_model_params["L1_Linear_b"], discrete_weights)
        s3_l1_repar = LocalReparametrization()  # outputs a value and not a dist.
        s4_l1_batchnorm = torch.nn.BatchNorm1d(1200, momentum=0.1)
        s5_l1_tanh = torch.nn.Tanh()

        s6_l2_dropout = torch.nn.Dropout(p=0.2)
        s7_l2_linear = LogitLinear(1200, ValueTypes.REAL, 1200, real_model_params["L2_Linear_W"],
                                   real_model_params["L2_Linear_b"], discrete_weights)
        s8_l2_repar = LocalReparametrization()  # outputs a value and not a dist.
        s9_l2_batchnorm = torch.nn.BatchNorm1d(1200, momentum=0.1)
        s10_l2_tanh = torch.nn.Tanh()

        s6_l3_dropout = torch.nn.Dropout(p=0.3)
        s7_l3_linear = LogitLinear(1200, ValueTypes.REAL, 10, real_model_params["L3_Linear_W"],
                                   real_model_params["L3_Linear_b"], discrete_weights, normalize_activations=True)
        s8_l3_repar = LocalReparametrization()
        # defining all the network's layers
        self.netlayers = torch.nn.Sequential(
            s1_l1_dropout,
            s2_l1_linear,
            s3_l1_repar,
            s4_l1_batchnorm,
            s5_l1_tanh,
            s6_l2_dropout,
            s7_l2_linear,
            s8_l2_repar,
            s9_l2_batchnorm,
            s10_l2_tanh,
            s6_l3_dropout,
            s7_l3_linear,
            s8_l3_repar)  # last layer outputs the unnormalized loglikelihood used by the softmax later

        """setting parameters of torch layers here"""
        new_state_dict = {
            'netlayers.3.weight': real_model_params["L1_BatchNorm_W"],
            'netlayers.3.bias': real_model_params["L1_BatchNorm_b"],
            'netlayers.8.weight': real_model_params["L2_BatchNorm_W"],
            'netlayers.8.bias': real_model_params["L2_BatchNorm_b"]
        }
        self.load_state_dict(new_state_dict, strict=False)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-10)
        print(self.optimizer)
        self.loss_funct = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # takes image vector
        return self.netlayers(x)

    def get_net_parameters(self):
        return self.state_dict()

    def set_net_parameters(self, param_dict):
        self.load_state_dict(param_dict, strict=False)

    def generate_discrete_networks(self, method: str) -> PiReal:
        """

        :param method: sample or argmax
        :return:
        """
        # state dicts
        l1_layer: LogitLinear = self.netlayers[1]
        l1_sampled_w, l1_sampled_b = l1_layer.generate_discrete_network(method)
        l2_layer: LogitLinear = self.netlayers[6]
        l2_sampled_w, l2_sampled_b = l2_layer.generate_discrete_network(method)
        l3_layer: LogitLinear = self.netlayers[11]
        l3_sampled_w, l3_sampled_b = l3_layer.generate_discrete_network(method)
        state_dict = {
            "L1_Linear_W": l1_sampled_w,
            "L1_Linear_b": l1_sampled_b,
            "L1_BatchNorm_W": self.state_dict()['netlayers.3.weight'],
            "L1_BatchNorm_b": self.state_dict()['netlayers.3.bias'],
            "L2_Linear_W": l2_sampled_w,
            "L2_Linear_b": l2_sampled_b,
            "L2_BatchNorm_W": self.state_dict()['netlayers.8.weight'],
            "L2_BatchNorm_b": self.state_dict()['netlayers.8.bias'],
            "L3_Linear_W": l3_sampled_w,
            "L3_Linear_b": l3_sampled_b
        }

        real_net = PiReal()
        real_net.set_net_parameters(state_dict)
        return real_net

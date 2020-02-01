import unittest

from discrete_nn.models.mnist_pi.real import MnistPiReal

class TestRealMnistPI(unittest.TestCase):
    def test_parameter_saving_loading(self):
        model = MnistPiReal()
        param = model.get_training_parameters()
        model.set_training_parameters(param)

if __name__ == '__main__':
    unittest.main()

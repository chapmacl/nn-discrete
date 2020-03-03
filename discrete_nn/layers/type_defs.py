from enum import Enum


class ValueTypes(Enum):
    REAL = "real"
    GAUSSIAN = "gaussian"

class DiscreteWeights(Enum):
    TERNARY = [-1, 0, 1]
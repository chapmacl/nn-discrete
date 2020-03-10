from enum import Enum


class ValueTypes(Enum):
    REAL = "real"
    GAUSSIAN = "gaussian"


class InputFormat(Enum):
    FLAT_ARRAY = "flat"
    FEATURE_MAP = "feature_maps"


class DiscreteWeights(Enum):
    TERNARY = [-1, 0, 1]

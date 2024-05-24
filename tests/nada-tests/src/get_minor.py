from nada_dsl import *
import numpy as np
import nada_algebra as na
from nada_algebra.array import NadaArray

LOG_SCALE = 16
SCALE = 1 << LOG_SCALE
PRIME_64 = 18446744072637906947
PRIME_128 = 340282366920938463463374607429104828419
PRIME_256 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF98C00003
PRIME = PRIME_64


def get_minor(matrix, i, j):
    """Return the Minor matrix after removing the i-th row and j-th column"""
    row_removed = matrix[:i].vstack(matrix[i + 1 :])
    return row_removed[:, :j].hstack(row_removed[:, j + 1 :])


def nada_main():
    parties = na.parties(3)

    X = na.array([3, 3], parties[0], "A", nada_type=SecretInteger)

    X = X.reveal()

    return get_minor(X, 0, 1).output(parties[2], "my_output")

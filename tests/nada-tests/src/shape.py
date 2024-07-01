from nada_dsl import *

import nada_numpy as na


def nada_main():
    parties = na.parties(3)

    a = na.array([3, 3], parties[0], "A", SecretInteger)

    result = na.zeros([3])
    for i in range(a.shape[1]):
        result += a[:, i]  # Sum by columns

    return result.output(parties[1], "my_output")

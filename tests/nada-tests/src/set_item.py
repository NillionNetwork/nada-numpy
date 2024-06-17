from nada_dsl import *

import nada_numpy as na


def nada_main():
    parties = na.parties(3)

    a = na.array([3, 3], parties[0], "A", SecretInteger)

    result = a[0, :] + a[1, :] + a[2, :]

    result[0] = result[0] + Integer(1)

    result[1] = result[0] + Integer(1)

    result[2] = result[1] + Integer(1)

    return result.output(parties[1], "my_output")

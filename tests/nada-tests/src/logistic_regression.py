from nada_dsl import *

import nada_algebra as na


def nada_main():
    parties = na.parties(3)

    weights = na.array((3,), parties[0], "A", SecretInteger)
    # This example does not support an array since we work with secret integers
    bias = SecretInteger(Input("bias", parties[0]))
    x = na.array((3,), parties[1], "B", SecretInteger)

    res = weights.dot(x)
    res += bias

    res = weights @ x
    res += bias

    return na.output(res, parties[2], "my_output")

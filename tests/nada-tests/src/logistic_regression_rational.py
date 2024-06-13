from nada_dsl import *

import nada_algebra as na


def nada_main():
    parties = na.parties(3)

    weights = na.array((3,), parties[0], "A", na.SecretRational)
    # bias = na.secret_rational("bias", parties[0])
    bias = na.array((1,), parties[0], "bias", na.SecretRational)
    x = na.array((3,), parties[1], "B", na.SecretRational)

    res = weights.dot(x)
    res += bias

    res = weights @ x
    res += bias

    return na.output(res, parties[2], "my_output")

from nada_dsl import *

import nada_numpy as na


def nada_main():
    parties = na.parties(3)

    a = na.array([3], parties[0], "A", SecretInteger)

    result = Integer(0)
    for i in range(a.shape[0]):  # GET ATTR
        result += a[i]  # GET ITEM

    return na.output(result, parties[1], "my_output")

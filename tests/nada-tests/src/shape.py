from nada_dsl import *

import nada_algebra as na


def nada_main():
    parties = na.parties(3)

    a = na.array([3, 3], parties[0], "A")

    result = na.zeros([3])
    for i in range(a.shape[1]):
        result += a[:, i]  # Sum by columns

    return result.output(parties[1], "my_output")

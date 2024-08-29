from nada_dsl import *

import nada_numpy as na


def nada_main():
    parties = na.parties(3)

    a = na.array([3], parties[0], "A", SecretInteger)
    b = na.array([3], parties[1], "B", SecretInteger)

    c = a.hstack(b)

    c += Integer(0)

    return c.output(parties[2], "my_output")

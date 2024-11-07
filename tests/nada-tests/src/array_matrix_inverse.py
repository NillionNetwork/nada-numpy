from nada_dsl import *

import nada_numpy as na


def nada_main():
    parties = na.parties(2)

    a = na.array([3, 3], parties[0], "A", SecretInteger)

    a_inv = a.inv()
    res = a @ a_inv 

    return res.output(parties[1], "my_output")

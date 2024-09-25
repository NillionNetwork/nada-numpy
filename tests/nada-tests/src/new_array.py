from nada_dsl import *

import nada_numpy as na


def nada_main():
    parties = na.parties(2)

    a = na.array([3, 3, 3], parties[0], "A", SecretInteger)

    b = a[0]
    c = a[0, 0]

    assert isinstance(a[0, 0, 0], SecretInteger), "a[0][0] should be a SecretInteger"
    assert isinstance(a[0], na.NadaArray), "a[0] should be a NadaArray"

    return a.output(parties[1], "my_output")

from nada_dsl import *

import nada_algebra as na


def nada_main():
    parties = na.parties(2)

    a = na.array([3], parties[0], "A", SecretInteger)

    return a.output(parties[1], "my_output")

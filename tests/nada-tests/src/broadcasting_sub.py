from nada_dsl import *

import nada_numpy as na


def nada_main():
    parties = na.parties(3)

    a = na.array([3], parties[0], "A", SecretInteger)
    b = na.array([3], parties[1], "B", SecretInteger)

    result1 = a - b
    result2 = a - Integer(2)

    return result1.output(parties[1], "my_output_1") + result2.output(
        parties[2], "my_output_2"
    )

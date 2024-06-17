from nada_dsl import *

import nada_numpy as na


def nada_main():
    parties = na.parties(3)

    a = na.array([3, 3], parties[0], "A", na.SecretRational)

    b = na.array([3, 3], parties[1], "B", na.SecretRational)

    c = na.array((3,), parties[2], "C", na.SecretRational)

    d = na.ones([3, 3], na.Rational)

    result_a = a @ b

    result_b = a @ c

    result_c = a @ d

    result_d = d @ a

    return (
        result_a.output(parties[1], "my_output")
        + result_b.output(parties[1], "my_output_b")
        + result_c.output(parties[1], "my_output_c")
        + result_d.output(parties[1], "my_output_d")
    )

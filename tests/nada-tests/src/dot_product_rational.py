from nada_dsl import *

import nada_numpy as na


def nada_main():
    parties = na.parties(3)

    a = na.array((3,), parties[0], "A", na.SecretRational)
    b = na.array((3,), parties[1], "B", na.SecretRational)
    c = na.ones((3,), na.Rational)

    result = a.dot(b)

    result_b = a @ b

    result_c = a.dot(c)

    result_d = a @ c

    return (
        na.output(result, parties[1], "my_output_a")
        + na.output(result_b, parties[1], "my_output_b")
        + na.output(result_c, parties[1], "my_output_c")
        + na.output(result_d, parties[1], "my_output_d")
    )

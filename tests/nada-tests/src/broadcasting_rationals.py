from nada_dsl import *

import nada_algebra as na


def nada_main():
    parties = na.parties(3)

    x = na.array((3,), parties[0], "X", na.SecretRational)
    y = na.secret_rational("y", parties[0])

    out_1 = x + y
    out_2 = y + x
    out_3 = x - y
    out_4 = y - x
    out_5 = x * y
    out_6 = y * x
    out_7 = x / y
    out_8 = y / x

    return (
        out_1.output(parties[2], "my_output_a")
        + out_2.output(parties[2], "my_output_b")
        + out_3.output(parties[2], "my_output_c")
        + out_4.output(parties[2], "my_output_d")
        + out_5.output(parties[2], "my_output_e")
        + out_6.output(parties[2], "my_output_f")
        + out_7.output(parties[2], "my_output_g")
        + out_8.output(parties[2], "my_output_h")
    )

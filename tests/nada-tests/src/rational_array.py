from nada_dsl import *
import nada_algebra as na


def nada_main():
    parties = na.parties(2)

    a = na.array([3], parties[0], "A", nada_type=na.SecretRational)
    b = na.array([3], parties[0], "B", nada_type=na.SecretRational)
    c = na.ones([3], na.Rational)

    out_0 = a + b
    out_1 = a - b
    out_2 = a * b
    out_3 = a / b

    out_4 = a + c
    out_5 = a - c
    out_6 = a * c
    out_7 = a / c

    return (
        out_0.output(parties[1], "out_0")
        + out_1.output(parties[1], "out_1")
        + out_2.output(parties[1], "out_2")
        + out_3.output(parties[1], "out_3")
        + out_4.output(parties[1], "out_4")
        + out_5.output(parties[1], "out_5")
        + out_6.output(parties[1], "out_6")
        + out_7.output(parties[1], "out_7")
    )

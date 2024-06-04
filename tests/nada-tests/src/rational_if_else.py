from nada_dsl import *
import nada_algebra as na


def nada_main():
    parties = na.parties(3)

    a = na.SecretRational("A", parties[0])
    b = na.SecretRational("B", parties[1])
    c = na.SecretRational("C", parties[2])

    out_0 = (a > b).if_else(na.Rational(0), na.Rational(1))
    out_1 = (a >= b).if_else(na.Rational(1), na.Rational(0))
    out_2 = (a < b).if_else(na.Rational(2), na.Rational(1))
    out_3 = (a <= b).if_else(na.Rational(3), na.Rational(0))

    out_4 = (a > b).if_else(c, na.Rational(2))
    out_5 = (a >= b).if_else(na.Rational(2), c)
    out_6 = (a < b).if_else(c, na.Rational(2))
    out_7 = (a <= b).if_else(c, na.Rational(2))

    return (
        na.output(out_0, parties[2], "out_0")
        + na.output(out_1, parties[2], "out_1")
        + na.output(out_2, parties[2], "out_2")
        + na.output(out_3, parties[2], "out_3")
        + na.output(out_4, parties[2], "out_4")
        + na.output(out_5, parties[2], "out_5")
        + na.output(out_6, parties[2], "out_6")
        + na.output(out_7, parties[2], "out_7")
    )

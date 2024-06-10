from nada_dsl import *
import nada_algebra as na


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
        result.output(parties[1], "my_output_a")
        + result_b.output(parties[1], "my_output_b")
        + result_c.output(parties[1], "my_output_c")
        + result_d.output(parties[1], "my_output_d")
    )

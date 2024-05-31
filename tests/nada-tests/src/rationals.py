from nada_dsl import *
import nada_algebra as na


def nada_main():
    parties = na.parties(3)

    a = na.array([3, 3], parties[0], "A", as_rational=True)
    b = Integer(2)

    result = a + b
    return na.output(result, parties[2], "my_output")

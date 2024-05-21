from nada_dsl import *
import nada_algebra as na


def nada_main():
    parties = na.parties(3)

    a = na.array([3], parties[0], "A")

    result = a.sum()

    return [Output(result, "my_output", parties[1])]

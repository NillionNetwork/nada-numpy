from nada_dsl import *
import nada_algebra as na


def nada_main():
    parties = na.parties(3)

    a = na.array([3], parties[0], "A", SecretInteger)
    b = na.array([3], parties[1], "B", SecretInteger)

    result = a.dot(b)

    return result.output(parties[1], "my_output")

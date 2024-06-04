from nada_dsl import *
import nada_algebra as na


def nada_main():
    parties = na.parties(3)

    a = na.array([3, 3, 3], parties[0], "A", SecretInteger)
    b = na.ones([3, 3, 3])
    c = na.zeros([3, 3, 3])
    d = na.ones([3, 3, 3])

    result = a + b + c - d

    return result.output(parties[1], "my_output")

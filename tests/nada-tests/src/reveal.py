from nada_dsl import *
import nada_algebra as na


def nada_main():
    parties = na.parties(3)

    a = na.array([3, 3], parties[0], "A")
    b = na.array([3, 3], parties[1], "B")
    c = a.reveal()

    result = b + c
    return na.output(result, parties[2], "my_output")

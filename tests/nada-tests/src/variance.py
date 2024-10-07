from nada_dsl import *
import nada_numpy as na

def nada_main():
    parties = na.parties(4)

    a = na.array([4], parties[0], "A", SecretInteger)

    a[0] = SecretInteger(Input("A_0", parties[0]))
    a[1] = SecretInteger(Input("A_1", parties[1]))
    a[2] = SecretInteger(Input("A_2", parties[2]))
    a[3] = SecretInteger(Input("A_3", parties[3]))

    var = a.var()

    return na.output(var, parties[1], "my_output")

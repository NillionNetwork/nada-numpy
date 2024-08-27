from types import NoneType

import numpy as np
from nada_dsl import *

import nada_numpy as na
from nada_numpy.nada_typing import NadaInteger, NadaRational


def nada_main():
    parties = na.parties(2)

    a = na.array([3], parties[0], "A", SecretInteger)
    b = na.array([3], parties[1], "B", SecretInteger)
    c = Integer(1)

    d = a == b
    e = a == c
    f = a != b
    g = a != c
    h = a < b
    i = a < c
    j = a <= b
    k = a <= c
    l = a > b
    m = a > c
    n = a >= b
    o = a >= c

    return (
        d.output(parties[1], "my_output_1")
        + e.output(parties[1], "my_output_2")
        + f.output(parties[1], "my_output_3")
        + g.output(parties[1], "my_output_4")
        + h.output(parties[1], "my_output_5")
        + i.output(parties[1], "my_output_6")
        + j.output(parties[1], "my_output_7")
        + k.output(parties[1], "my_output_8")
        + l.output(parties[1], "my_output_9")
        + m.output(parties[1], "my_output_10")
        + n.output(parties[1], "my_output_11")
        + o.output(parties[1], "my_output_12")
    )

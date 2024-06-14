from types import NoneType

import numpy as np
from nada_dsl import *

import nada_algebra as na
from nada_algebra.nada_typing import NadaInteger, NadaRational


def nada_main():
    parties = na.parties(2)

    a = na.array([3], parties[0], "A", SecretInteger)
    b = na.NadaArray(np.array([]))
    c = na.NadaArray(np.array([na.rational(1.5)]))

    assert not a.empty
    assert b.empty
    assert not c.empty

    assert a.dtype == NadaInteger, a.dtype
    assert b.dtype == None, b.dtype
    assert c.dtype == NadaRational, c.dtype

    assert a.ndim == 1, a.ndim
    assert b.ndim == 1, b.ndim
    assert c.ndim == 1, c.ndim

    assert len(a) == 3, len(a)
    assert len(b) == 0, len(b)
    assert len(c) == 1, len(c)

    return a.output(parties[1], "my_output")

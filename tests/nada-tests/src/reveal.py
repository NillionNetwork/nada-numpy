import pytest
from nada_dsl import *

import nada_numpy as na
from nada_numpy.nada_typing import NadaInteger, NadaRational


def nada_main():
    parties = na.parties(3)

    a = na.array([3, 3], parties[0], "A", SecretInteger)
    b = na.array([3, 3], parties[1], "B", na.SecretRational)

    c = a.reveal()
    d = b.reveal()

    assert c.dtype == NadaInteger, c.dtype
    assert c.shape == a.shape

    assert d.dtype == NadaRational, d.dtype
    assert d.shape == b.shape

    with pytest.raises(Exception):
        c.reveal()
    with pytest.raises(Exception):
        d.reveal()

    return na.output(c, parties[2], "my_output_A") + na.output(
        d, parties[2], "my_output_B"
    )

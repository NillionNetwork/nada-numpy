import pytest
from nada_dsl import *
import nada_algebra as na


def nada_main():
    parties = na.parties(2)

    na.set_log_scale(32)

    a = na.array([3], parties[0], "A", na.SecretRational)

    b = a + na.rational(2)

    na.reset_log_scale()

    with pytest.raises(ValueError):
        b + na.rational(2)

    with pytest.warns():
        na.set_log_scale(2**16)

    with pytest.warns():
        na.set_log_scale(0)

    return b.output(parties[1], "my_output")

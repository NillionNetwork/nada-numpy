import pytest
from nada_dsl import *

import nada_algebra as na


def nada_main():
    parties = na.parties(2)

    na.set_log_scale(32)

    a = na.array([3], parties[0], "A", na.SecretRational)

    b = a + na.rational(2)  # both values are on scale 32

    na.reset_log_scale()  # resets the log scale back to the original default (16)

    with pytest.raises(ValueError):
        b + na.rational(2)  # scale 32 rational + scale 16 rational

    with pytest.warns():
        na.set_log_scale(2**16)  # extremely high - most likely a mistake

    with pytest.warns():
        na.set_log_scale(0)  # extremely low - most likely a mistake

    return b.output(parties[1], "my_output")

import pytest
from nada_dsl import *

import nada_numpy as na


def nada_main():
    parties = na.parties(2)

    a = na.array([3], parties[0], "A", na.SecretRational)
    b = na.array([3], parties[0], "B", na.Rational)
    c = na.array([3], parties[0], "B", SecretInteger)
    d = na.array([3], parties[0], "B", PublicInteger)

    with pytest.raises(TypeError):
        a + c
    with pytest.raises(TypeError):
        a + d
    with pytest.raises(TypeError):
        b + c
    with pytest.raises(TypeError):
        b + d
    with pytest.raises(TypeError):
        c + a
    with pytest.raises(TypeError):
        d + a
    with pytest.raises(TypeError):
        c + b
    with pytest.raises(TypeError):
        d + b

    result = a + b

    return result.output(parties[1], "my_output")

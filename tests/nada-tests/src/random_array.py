import pytest
from nada_dsl import *

import nada_algebra as na


def nada_main():
    parties = na.parties(2)

    a = na.array([3], parties[0], "A", SecretInteger)

    supported_types = [SecretInteger, SecretUnsignedInteger, na.SecretRational]

    for supported_type in supported_types:
        random_arr_1 = na.random((1,), supported_type)
        assert random_arr_1.shape == (1,), random_arr_1.shape
        assert isinstance(random_arr_1.item(0), supported_type)

        random_arr_2 = na.random((4, 2, 3), supported_type)
        assert random_arr_2.shape == (4, 2, 3), random_arr_2.shape
        assert isinstance(random_arr_2.item(0), supported_type)

    with pytest.raises(Exception):
        na.random((1,), PublicInteger)

    with pytest.raises(Exception):
        na.random((1,), na.Rational)

    return a.output(parties[1], "my_output")

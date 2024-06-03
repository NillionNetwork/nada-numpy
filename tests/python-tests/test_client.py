"""Nada algebra client unit tests"""

import numpy as np
import nada_algebra.client as na_client
import nada_algebra as na
import py_nillion_client as nillion


class TestClient:

    def test_array_1(self):
        input_arr = np.random.randn(1)
        nada_array = na_client.array(input_arr, "test")

        assert list(nada_array.keys()) == ["test_0"]

    def test_array_2(self):
        input_arr = np.random.randn(1)
        nada_array = na_client.array(input_arr, "test", na.Rational)

        assert list(nada_array.keys()) == ["test_0"]

    def test_array_3(self):
        input_arr = np.random.randn(1)
        nada_array = na_client.array(input_arr, "test", na.SecretRational)

        assert list(nada_array.keys()) == ["test_0"]

    def test_array_4(self):
        input_arr = np.random.randn(1)
        nada_array = na_client.array(input_arr, "test", nillion.PublicVariableInteger)

        assert list(nada_array.keys()) == ["test_0"]

    def test_array_5(self):
        input_arr = np.random.randn(3)
        nada_array = na_client.array(input_arr, "test")

        assert list(nada_array.keys()) == ["test_0", "test_1", "test_2"]

    def test_array_6(self):
        input_arr = np.random.randn(2, 3)
        nada_array = na_client.array(input_arr, "test")

        assert list(sorted(nada_array.keys())) == [
            "test_0_0",
            "test_0_1",
            "test_0_2",
            "test_1_0",
            "test_1_1",
            "test_1_2",
        ]

    def test_array_7(self):
        input_arr = np.array([])
        nada_array = na_client.array(input_arr, "test")

        assert nada_array == {}

    def test_concat(self):
        dict_1 = {"a": 1, "b": 2}
        dict_2 = {"c": 3}

        dict_comb = na_client.concat([dict_1, dict_2])

        assert dict_comb == {"a": 1, "b": 2, "c": 3}

    def test_secret_rational_1(self):
        test_value = 1

        rational = na_client.SecretRational(test_value)

        assert isinstance(rational, nillion.SecretInteger)

        rational_value = rational.value

        assert rational_value == test_value * 2**na.types.RationalConfig.LOG_SCALE

    def test_secret_rational_2(self):
        test_value = 2.5

        rational = na_client.SecretRational(test_value)

        assert isinstance(rational, nillion.SecretInteger)

        rational_value = rational.value

        assert rational_value == test_value * 2**na.types.RationalConfig.LOG_SCALE

    def test_public_rational_1(self):
        test_value = 1

        rational = na_client.PublicRational(test_value)

        assert isinstance(rational, nillion.PublicVariableInteger)

        rational_value = rational.value

        assert rational_value == test_value * 2**na.types.RationalConfig.LOG_SCALE

    def test_public_rational_2(self):
        test_value = 2.5

        rational = na_client.PublicRational(test_value)

        assert isinstance(rational, nillion.PublicVariableInteger)

        rational_value = rational.value

        assert rational_value == test_value * 2**na.types.RationalConfig.LOG_SCALE

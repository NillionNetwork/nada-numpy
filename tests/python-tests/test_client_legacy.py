"""Nada numpy client unit tests"""

import numpy as np
import py_nillion_client as nillion
import pytest

import nada_numpy as na
import nada_numpy.legacy_client as na_client


class TestClient:

    def test_array_1(self):
        input_arr = np.array([1])
        nada_array = na_client.array(input_arr, "test", na.Rational)

        assert list(nada_array.keys()) == ["test_0"]

    def test_array_2(self):
        input_arr = np.array([0])
        nada_array = na_client.array(input_arr, "test", na.SecretRational)

        assert list(nada_array.keys()) == ["test_0"]

    def test_array_3(self):
        input_arr = np.array([0])

        nada_array_1 = na_client.array(input_arr, "test", nillion.SecretInteger)
        assert isinstance(
            nada_array_1["test_0"], nillion.SecretInteger
        ), f"SecretInteger should be SecretInteger but got {type(nada_array_1['test_0'])}"

        nada_array_2 = na_client.array(input_arr, "test", nillion.SecretUnsignedInteger)
        assert isinstance(
            nada_array_2["test_0"], nillion.SecretUnsignedInteger
        ), f"SecretUnsignedInteger should be SecretUnsignedInteger but got {type(nada_array_2['test_0'])}"

        nada_array_3 = na_client.array(input_arr, "test", nillion.Integer)
        assert isinstance(
            nada_array_3["test_0"], nillion.Integer
        ), f"Integer should be Integer but got {type(nada_array_3['test_0'])}"

        nada_array_4 = na_client.array(input_arr, "test", nillion.Integer)
        assert isinstance(
            nada_array_4["test_0"], nillion.Integer
        ), f"Integer should be Integer but got {type(nada_array_4['test_0'])}"

        nada_array_5 = na_client.array(input_arr, "test", na.Rational)
        assert isinstance(
            nada_array_5["test_0"], nillion.Integer
        ), f"Rational should be Integer but got {type(nada_array_5['test_0'])}"

        nada_array_6 = na_client.array(input_arr, "test", na.SecretRational)
        assert (
            type(nada_array_6["test_0"]) == nillion.SecretInteger
        ), f"SecretRational should be SecretInteger but got {type(nada_array_6['test_0'])}"
        assert isinstance(
            nada_array_6["test_0"], nillion.SecretInteger
        ), f"SecretRational should be SecretInteger but got {type(nada_array_6['test_0'])}"

    def test_array_4(self):
        input_arr = np.array([0])
        nada_array = na_client.array(input_arr, "test", nillion.Integer)

        assert list(nada_array.keys()) == ["test_0"]

    def test_array_5(self):
        input_arr = np.random.randn(3)
        nada_array = na_client.array(input_arr, "test", nillion.SecretInteger)

        assert list(nada_array.keys()) == ["test_0", "test_1", "test_2"]

    def test_array_6(self):
        input_arr = np.random.randn(2, 3)
        nada_array = na_client.array(input_arr, "test", nillion.SecretInteger)

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

        supported_types = [
            nillion.SecretInteger,
            nillion.SecretUnsignedInteger,
            nillion.Integer,
            nillion.UnsignedInteger,
            na.Rational,
            na.SecretRational,
        ]
        for supported_type in supported_types:
            nada_array = na_client.array(input_arr, "test", supported_type)
            assert nada_array == {}

    def test_array_8(self):
        with pytest.raises(TypeError):
            na_client.array(np.array([1]), "my_array")  # no nada_type provided

    def test_concat_1(self):
        dict_1 = {"a": 1, "b": 2}
        dict_2 = {"c": 3}

        dict_comb = na_client.concat([dict_1, dict_2])

        assert dict_comb == {"a": 1, "b": 2, "c": 3}

    def test_concat_2(self):
        dict_1 = {"a": 1, "b": 2}
        dict_2 = {"b": 3}

        dict_comb = na_client.concat([dict_1, dict_2])

        assert dict_comb == {"a": 1, "b": 3}

    def test_secret_rational_1(self):
        test_value = 1

        rational = na_client.secret_rational(test_value)

        assert isinstance(rational, nillion.SecretInteger)

        rational_value = rational.value

        assert rational_value == test_value * 2 ** na.get_log_scale()

    def test_secret_rational_2(self):
        test_value = 2.5

        rational = na_client.secret_rational(test_value)

        assert isinstance(rational, nillion.SecretInteger)

        rational_value = rational.value

        assert rational_value == test_value * 2 ** na.get_log_scale()

    def test_public_rational_1(self):
        test_value = 1

        rational = na_client.public_rational(test_value)

        assert isinstance(rational, nillion.Integer)

        rational_value = rational.value

        assert rational_value == test_value * 2 ** na.get_log_scale()

    def test_public_rational_2(self):
        test_value = 2.5

        rational = na_client.public_rational(test_value)

        assert isinstance(rational, nillion.Integer)

        rational_value = rational.value

        assert rational_value == test_value * 2 ** na.get_log_scale()

    def test_parties_1(self):
        parties = na_client.parties(2)

        assert list(sorted(parties)) == ["Party0", "Party1"]

    def test_parties_2(self):
        parties = na_client.parties(0)

        assert parties == []

    def test_parties_3(self):
        parties = na_client.parties(3, "my_party")

        assert parties == ["my_party0", "my_party1", "my_party2"]

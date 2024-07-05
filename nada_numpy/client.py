"""
This module provides functions to work with the Python Nillion Client for handling
secret and public variable integers and generating named party objects and input dictionaries.
"""

from typing import Dict, List, Optional, Union

import numpy as np
# pylint:disable=no-name-in-module
from py_nillion_client import (Integer, SecretInteger, SecretUnsignedInteger,
                               UnsignedInteger)

from nada_numpy.types import Rational, SecretRational, get_log_scale

__all__ = [
    "parties",
    "array",
    "concat",
    "public_rational",
    "secret_rational",
    "float_from_rational",
]


def parties(num: int, prefix: str = "Party") -> List:
    """
    Creates a list of party name strings.

    Args:
        num (int): The number of parties to create.
        prefix (str, optional): The prefix to use for party names. Defaults to "Party".

    Returns:
        List: A list of party name strings in the format "{prefix}{i}".
    """
    return [f"{prefix}{i}" for i in range(num)]


def array(
    arr: np.ndarray,
    prefix: str,
    nada_type: Union[
        SecretInteger,
        SecretUnsignedInteger,
        Integer,
        UnsignedInteger,
        Rational,
        SecretRational,
    ],
) -> Dict:
    """
    Recursively generates a dictionary of Nillion input objects for each element
    in the given array.

    Args:
        arr (np.ndarray): The input array.
        prefix (str): The prefix to be added to the output names.
        nada_type (type): The type of the values introduced.

    Returns:
        Dict: A dictionary mapping generated names to Nillion input objects.
    """
    # TODO: Use  this version when check for zero values is removed
    # if len(arr.shape) == 1:
    #     if nada_type == Rational:
    #         nada_type = public_rational  # type: ignore
    #     elif nada_type == SecretRational:
    #         nada_type = secret_rational  # type: ignore
    #     return {
    #         f"{prefix}_{i}": (nada_type(int(arr[i]))) for i in range(arr.shape[0])  # type: ignore
    #     }

    # TODO: remove check for zero values when pushing zero secrets is supported

    if len(arr.shape) == 1:
        if nada_type == Rational:
            return {
                f"{prefix}_{i}": (public_rational(arr[i])) for i in range(arr.shape[0])
            }
        if nada_type == SecretRational:
            return {
                f"{prefix}_{i}": (
                    secret_rational(arr[i]) if arr[i] != 0 else SecretInteger(1)
                )
                for i in range(arr.shape[0])
            }
        return {
            f"{prefix}_{i}": (
                nada_type(int(arr[i]))  # type: ignore
                if (nada_type in (Integer, UnsignedInteger) or int(arr[i]) != 0)
                else nada_type(1)  # type: ignore
            )
            for i in range(arr.shape[0])
        }
    return {
        k: v
        for i in range(arr.shape[0])
        for k, v in array(arr[i], f"{prefix}_{i}", nada_type).items()
    }


def concat(list_dict: List[Dict]) -> Dict:
    """
    Combines a list of dictionaries into a single dictionary.

    Note: This function will overwrite values for duplicate keys.

    Args:
        list_dict (List[Dict]): A list of dictionaries.

    Returns:
        Dict: A single merged dictionary.
    """
    return {k: v for d in list_dict for k, v in d.items()}


def __rational(value: Union[float, int]) -> int:
    """
    Returns the integer representation of the given float value.

    Args:
        value (Union[float, int]): The input value.

    Returns:
        int: The integer representation of the input value.
    """
    return round(value * (1 << get_log_scale()))


def public_rational(value: Union[float, int]) -> Integer:
    """
    Returns the integer representation of the given float value.

    Args:
        value (Union[float, int]): The input value.

    Returns:
        int: The integer representation of the input value.
    """
    return Integer(__rational(value))


def secret_rational(value: Union[float, int]) -> SecretInteger:
    """
    Returns the integer representation of the given float value.

    Args:
        value (Union[float, int]): The input value.

    Returns:
        int: The integer representation of the input value.
    """
    return SecretInteger(__rational(value))


def float_from_rational(value: int, log_scale: Optional[int] = None) -> float:
    """
    Returns the float representation of the given rational value.

    Args:
        value (int): The output Rational value to convert.
        log_scale (int, optional): The log scale to use for conversion. Defaults to None.

    Returns:
        float: The float representation of the input value.
    """
    if log_scale is None:
        log_scale = get_log_scale()
    return value / (1 << log_scale)

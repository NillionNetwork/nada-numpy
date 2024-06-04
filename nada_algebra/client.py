"""
This module provides functions to work with the Python Nillion Client for handling
secret and public variable integers and generating named party objects and input dictionaries.
"""

from typing import Union
from py_nillion_client import (
    SecretInteger,
    SecretUnsignedInteger,
    PublicVariableInteger,
    PublicVariableUnsignedInteger,
)
import numpy as np
from nada_algebra import types


def parties(num: int, prefix: str = "Party") -> list:
    """
    Creates a list of party name strings.

    Args:
        num (int): The number of parties to create.
        prefix (str, optional): The prefix to use for party names. Defaults to "Party".

    Returns:
        list: A list of party name strings in the format "{prefix}{i}".
    """
    return [f"{prefix}{i}" for i in range(num)]


def array(
    arr: np.ndarray,
    prefix: str,
    nada_type: Union[
        SecretInteger,
        SecretUnsignedInteger,
        PublicVariableInteger,
        PublicVariableUnsignedInteger,
        types.Rational,
        types.SecretRational,
    ] = SecretInteger,
) -> dict:
    """
    Recursively generates a dictionary of Nillion input objects for each element
    in the given array.

    Args:
        arr (np.ndarray): The input array.
        prefix (str): The prefix to be added to the output names.
        nada_type (type, optional): The type of the values introduced. Defaults to SecretInteger.

    Returns:
        dict: A dictionary mapping generated names to Nillion input objects.
    """
    # TODO: remove check for zero values when pushing zero secrets is supported
    if len(arr.shape) == 1:
        if nada_type == types.Rational:
            return {
                f"{prefix}_{i}": (PublicRational(arr[i])) for i in range(arr.shape[0])
            }
        if nada_type == types.SecretRational:
            return {
                f"{prefix}_{i}": (
                    SecretRational(arr[i]) if arr[i] != 0 else SecretInteger(1)
                )
                for i in range(arr.shape[0])
            }
        return {
            f"{prefix}_{i}": (
                nada_type(int(arr[i]))
                if (
                    nada_type in (PublicVariableInteger, PublicVariableUnsignedInteger)
                    or int(arr[i]) != 0
                )
                else nada_type(1)
            )
            for i in range(arr.shape[0])
        }
    return {
        k: v
        for i in range(arr.shape[0])
        for k, v in array(arr[i], f"{prefix}_{i}", nada_type).items()
    }


def concat(list_dict: list[dict]) -> dict:
    """
    Combines a list of dictionaries into a single dictionary.

    Note: This function will overwrite values for duplicate keys.

    Args:
        list_dict (list[dict]): A list of dictionaries.

    Returns:
        dict: A single merged dictionary.
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
    return round(value * (1 << types.RationalConfig.LOG_SCALE))


def PublicRational(value: Union[float, int]) -> PublicVariableInteger:
    """
    Returns the integer representation of the given float value.

    Args:
        value (Union[float, int]): The input value.

    Returns:
        int: The integer representation of the input value.
    """
    return PublicVariableInteger(__rational(value))


def SecretRational(value: Union[float, int]) -> SecretInteger:
    """
    Returns the integer representation of the given float value.

    Args:
        value (Union[float, int]): The input value.

    Returns:
        int: The integer representation of the input value.
    """
    return SecretInteger(__rational(value))

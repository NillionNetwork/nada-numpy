"""
This module provides functions to work with the Python Nillion Client for handling
secret and public variable integers and generating named party objects and input dictionaries.
"""

from py_nillion_client import (
    SecretInteger,
    SecretUnsignedInteger,
    PublicVariableInteger,
    PublicVariableUnsignedInteger,
)
import numpy as np


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
    nada_type: (
        type(SecretInteger)
        | type(SecretUnsignedInteger)
        | type(PublicVariableInteger)
        | type(PublicVariableUnsignedInteger)
    ) = SecretInteger,
) -> dict:
    """
    Recursively generates a dictionary of Nillion input objects for each element
    in the given array.

    Args:
        arr (np.ndarray): The input array.
        prefix (str): The prefix to be added to the output names.
        nada_type (Union[type[SecretInteger], type[SecretUnsignedInteger], \
            type[PublicVariableInteger], type[PublicVariableUnsignedInteger]], optional):
            The type of the values introduced. Defaults to SecretInteger.

    Returns:
        dict: A dictionary mapping generated names to Nillion input objects.
    """
    if len(arr.shape) == 1:
        return {f"{prefix}_{i}": nada_type(int(arr[i])) for i in range(arr.shape[0])}
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

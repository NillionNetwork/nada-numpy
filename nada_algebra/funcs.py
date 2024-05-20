"""
This module provides common functions to work with Nada Algebra, including the creation
and manipulation of arrays and party objects.
"""

from nada_dsl import (
    Party,
    SecretInteger,
    SecretUnsignedInteger,
    PublicInteger,
    PublicUnsignedInteger,
    Integer,
    UnsignedInteger,
)
import numpy as np
from nada_algebra.array import NadaArray


def parties(num: int, prefix: str = "Party") -> list:
    """
    Create a list of Party objects with specified names.

    Args:
        num (int): The number of parties to create.
        prefix (str, optional): The prefix to use for party names. Defaults to "Party".

    Returns:
        list: A list of Party objects with names in the format "{prefix}{i}".
    """
    return [Party(name=f"{prefix}{i}") for i in range(num)]


def __from_list(lst: list, nada_type: Integer | UnsignedInteger) -> list:
    """
    Recursively convert a nested list to a list of NadaInteger objects.

    Args:
        lst (list): A nested list of integers.
        nada_type (type): The type of NadaInteger objects to create.

    Returns:
        list: A nested list of NadaInteger objects.
    """
    if len(lst.shape) == 1:
        return [nada_type(int(elem)) for elem in lst]
    return [__from_list(lst[i], nada_type) for i in range(len(lst))]


def from_list(lst: list, nada_type: Integer | UnsignedInteger = Integer) -> NadaArray:
    """
    Create a cleartext NadaArray from a list of integers.

    Args:
        lst (list): A list of integers representing the elements of the array.
        nada_type (type, optional): The type of NadaInteger objects to create. Defaults to Integer.

    Returns:
        NadaArray: The created NadaArray.
    """
    if not isinstance(lst, np.ndarray):
        lst = np.array(lst)
    return NadaArray(np.array(__from_list(lst, nada_type)))


def ones(dims: list, nada_type: Integer | UnsignedInteger = Integer) -> NadaArray:
    """
    Create a cleartext NadaArray filled with ones.

    Args:
        dims (list): A list of integers representing the dimensions of the array.
        nada_type (type, optional): The type of NadaInteger objects to create. Defaults to Integer.

    Returns:
        NadaArray: The created NadaArray filled with ones.
    """
    return from_list(np.ones(dims), nada_type)


def zeros(dims: list, nada_type: Integer | UnsignedInteger = Integer) -> NadaArray:
    """
    Create a cleartext NadaArray filled with zeros.

    Args:
        dims (list): A list of integers representing the dimensions of the array.
        nada_type (type, optional): The type of NadaInteger objects to create. Defaults to Integer.

    Returns:
        NadaArray: The created NadaArray filled with zeros.
    """
    return from_list(np.zeros(dims), nada_type)


def array(
    dims: list,
    party: Party,
    prefix: str,
    nada_type: (
        SecretInteger | SecretUnsignedInteger | PublicInteger | PublicUnsignedInteger
    ) = SecretInteger,
) -> NadaArray:
    """
    Create a NadaArray with the specified dimensions and elements of the given type.

    Args:
        dims (list): A list of integers representing the dimensions of the array.
        party (Party): The party object.
        prefix (str): A prefix for naming the array elements.
        nada_type (type, optional): The type of elements to create. Defaults to SecretInteger.

    Returns:
        NadaArray: The created NadaArray.
    """
    return NadaArray.array(dims, party, prefix, nada_type)


def random(
    dims: list, nada_type: SecretInteger | SecretUnsignedInteger = SecretInteger
) -> NadaArray:
    """
    Create a random NadaArray with the specified dimensions.

    Args:
        dims (list): A list of integers representing the dimensions of the array.
        nada_type (type, optional): The type of elements to create. Defaults to SecretInteger.

    Returns:
        NadaArray: A NadaArray with random values of the specified type.
    """
    return NadaArray.random(dims, nada_type)


def output(arr: NadaArray, party: Party, prefix: str):
    """
    Generate a list of Output objects for each element in the input NadaArray.

    Args:
        arr (NadaArray): The input NadaArray.
        party (Party): The party object.
        prefix (str): The prefix for naming the Output objects.

    Returns:
        list: A list of Output objects.
    """
    return arr.output(party, prefix)

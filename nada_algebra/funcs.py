"""
This module provides common functions to work with Nada Algebra, including the creation
and manipulation of arrays and party objects.
"""

from typing import Any, Callable, Iterable, Union
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
from nada_algebra.types import Rational, SecretRational, rational


_NadaCleartextType = Union[Integer, UnsignedInteger, Rational]


# These functions take at least a NadaArray argument and
# get referred to the NadaArray method with the same name
SUPPORTED_FUNCTIONAL_OPS = [
    "compress",
    "copy",
    "cumprod",
    "cumsum",
    "diagonal",
    "ndim",
    "prod",
    "put",
    "ravel",
    "repeat",
    "reshape",
    "resize",
    "shape",
    "size",
    "squeeze",
    "sum",
    "swapaxes",
    "take",
    "trace",
    "transpose",
]


def __create_func(func_name: str) -> Callable[..., Any]:
    """
    Creates a function with a given name.

    Args:
        func_name (str): Given function name.

    Returns:
        Callable[..., Any]: Created function object.
    """

    def func(a: NadaArray, *args, **kwargs) -> Any:
        """
        Function that takes at least a NadaArray input and returns whatever output
        the NadaArray method with the same name would have returned.

        E.g.,: `some_func(nada_array, arg_0, arg1=arg_1)` is made equivalent to
        `nada_array.some_func(arg_0, arg_1=arg_1)` as is the case in NumPy.

        Args:
            a (NadaArray): NadaArray input.

        Raises:
            TypeError: Raised when input array is not of type `NadaArray`.

        Returns:
            Any: Some output.
        """
        if not isinstance(a, NadaArray):
            raise TypeError(
                "Function operations %s requires input array of type NadaArray but received `%s`"
                % (func_name, type(a).__name__)
            )
        return getattr(a, func_name)(*args, **kwargs)

    return func

# Refers any functional call to the corresponding method call
for func_name in SUPPORTED_FUNCTIONAL_OPS:
    globals()[func_name] = __create_func(func_name)


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


def __from_numpy(arr: np.ndarray, nada_type: _NadaCleartextType) -> list:
    """
    Recursively convert a n-dimensional NumPy array to a nested list of NadaInteger objects.

    Args:
        arr (np.ndarray): A NumPy array of integers.
        nada_type (type): The type of NadaInteger objects to create.

    Returns:
        list: A nested list of NadaInteger objects.
    """
    if len(arr.shape) == 1:
        if isinstance(nada_type, Rational):
            return [nada_type(elem) for elem in arr]
        return [nada_type(int(elem)) for elem in arr]
    return [__from_numpy(arr[i], nada_type) for i in range(arr.shape[0])]


def from_list(lst: list, nada_type: _NadaCleartextType = Integer) -> NadaArray:
    """
    Create a cleartext NadaArray from a list of integers.

    Args:
        lst (list): A list of integers representing the elements of the array.
        nada_type (type, optional): The type of NadaInteger objects to create. Defaults to Integer.

    Returns:
        NadaArray: The created NadaArray.
    """
    if nada_type == Rational:
        nada_type = rational
    if not isinstance(lst, np.ndarray):
        lst = np.array(lst)
    return NadaArray(np.array(__from_numpy(lst, nada_type)))


def ones(dims: Iterable[int], nada_type: _NadaCleartextType = Integer) -> NadaArray:
    """
    Create a cleartext NadaArray filled with ones.

    Args:
        dims (Iterable[int]): A list of integers representing the dimensions of the array.
        nada_type (type, optional): The type of NadaInteger objects to create. Defaults to Integer.

    Returns:
        NadaArray: The created NadaArray filled with ones.
    """
    if nada_type == Rational:
        nada_type = rational
    return from_list(np.ones(dims), nada_type)


def ones_like(
    a: np.ndarray | NadaArray, nada_type: _NadaCleartextType = Integer
) -> NadaArray:
    """
    Create a cleartext NadaArray filled with one with the same shape and type as a given array.

    Args:
        a (np.ndarray | NadaArray): A reference array.
        nada_type (type, optional): The type of NadaInteger objects to create. Defaults to Integer.

    Returns:
        NadaArray: The created NadaArray filled with ones.
    """
    if nada_type == Rational:
        nada_type = rational
    if isinstance(a, NadaArray):
        a = a.inner
    return from_list(np.ones_like(a), nada_type)


def zeros(dims: Iterable[int], nada_type: _NadaCleartextType = Integer) -> NadaArray:
    """
    Create a cleartext NadaArray filled with zeros.

    Args:
        dims (Iterable[int]): A list of integers representing the dimensions of the array.
        nada_type (type, optional): The type of NadaInteger objects to create. Defaults to Integer.

    Returns:
        NadaArray: The created NadaArray filled with zeros.
    """
    if nada_type == Rational:
        nada_type = rational
    return from_list(np.zeros(dims), nada_type)


def zeros_like(
    a: np.ndarray | NadaArray, nada_type: _NadaCleartextType = Integer
) -> NadaArray:
    """
    Create a cleartext NadaArray filled with zeros with the same shape and type as a given array.

    Args:
        a (np.ndarray | NadaArray): A reference array.
        nada_type (type, optional): The type of NadaInteger objects to create. Defaults to Integer.

    Returns:
        NadaArray: The created NadaArray filled with zeros.
    """
    if nada_type == Rational:
        nada_type = rational
    if isinstance(a, NadaArray):
        a = a.inner
    return from_list(np.zeros_like(a), nada_type)


def alphas(dims: Iterable[int], alpha: Any) -> NadaArray:
    """
    Create a NadaArray filled with a certain constant value.

    Args:
        dims (Iterable[int]): A list of integers representing the dimensions of the array.
        alpha (Any): Some constant value.

    Returns:
        NadaArray: NadaArray filled with constant value.
    """
    ones_array = np.ones(dims)
    return NadaArray(np.frompyfunc(lambda _: alpha, 1, 1)(ones_array))


def alphas_like(a: np.ndarray | NadaArray, alpha: Any) -> NadaArray:
    """
    Create a NadaArray filled with a certain constant value with the same shape and type as a given array.

    Args:
        a (np.ndarray | NadaArray): Reference array.
        alpha (Any): Some constant value.

    Returns:
        NadaArray: NadaArray filled with constant value.
    """
    if isinstance(a, NadaArray):
        a = a.inner
    ones_array = np.ones_like(a)
    return NadaArray(np.frompyfunc(lambda _: alpha, 1, 1)(ones_array))


def array(
    dims: Iterable[int],
    party: Party,
    prefix: str,
    nada_type: Union[
        SecretInteger,
        SecretUnsignedInteger,
        PublicInteger,
        PublicUnsignedInteger,
        SecretRational,
        Rational,
    ],
) -> NadaArray:
    """
    Create a NadaArray with the specified dimensions and elements of the given type.

    Args:
        dims (Iterable[int]): A list of integers representing the dimensions of the array.
        party (Party): The party object.
        prefix (str): A prefix for naming the array elements.
        nada_type (type): The type of elements to create.

    Returns:
        NadaArray: The created NadaArray.
    """
    return NadaArray.array(dims, party, prefix, nada_type)


def random(
    dims: Iterable[int],
    nada_type: SecretInteger | SecretUnsignedInteger | SecretRational = SecretInteger,
) -> NadaArray:
    """
    Create a random NadaArray with the specified dimensions.

    Args:
        dims (Iterable[int]): A list of integers representing the dimensions of the array.
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
    return NadaArray.output_array(arr, party, prefix)


def vstack(arr_list: list) -> NadaArray:
    """
    Stack arrays in sequence vertically (row wise).

    Args:
        arr_list (list): A list of NadaArray objects to stack.

    Returns:
        NadaArray: The stacked NadaArray.
    """
    return NadaArray(np.vstack(arr_list))


def hstack(arr_list: list) -> NadaArray:
    """
    Stack arrays in sequence horizontally (column wise).

    Args:
        arr_list (list): A list of NadaArray objects to stack.

    Returns:
        NadaArray: The stacked NadaArray.
    """
    return NadaArray(np.hstack(arr_list))

__all__ = [
    name for name, obj in globals().items()
    if callable(obj) and not name.startswith('_')
] + SUPPORTED_FUNCTIONAL_OPS

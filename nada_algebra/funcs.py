"""
This module provides common functions to work with Nada Algebra, including the creation
and manipulation of arrays and party objects.
"""

from typing import Any, Iterable, Tuple, Union
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
from nada_algebra.utils import copy_metadata


_NadaCleartextType = Union[Integer, UnsignedInteger, Rational]


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


def empty(arr: NadaArray) -> bool:
    """
    Returns whether provided array is empty or not.

    Args:
        arr (NadaArray): Input array.

    Returns:
        bool: Whether array is empty or not.
    """
    return arr.empty


def ndim(arr: NadaArray) -> int:
    """
    Returns number of array dimensions.

    Args:
        arr (NadaArray): Input array.

    Returns:
        bool: Array dimensions.
    """
    return arr.ndim


def shape(arr: NadaArray) -> Tuple[int]:
    """
    Returns Array shape.

    Args:
        arr (NadaArray): Input array.

    Returns:
        bool: Array shape.
    """
    return arr.shape


def size(arr: NadaArray) -> int:
    """
    Returns array size.

    Args:
        arr (NadaArray): Input array.

    Returns:
        bool: Array size.
    """
    return arr.size


def to_nada(arr: np.ndarray, nada_type: _NadaCleartextType) -> NadaArray:
    """
    Converts a plain-text NumPy array to the equivalent NadaArray with
    a specified compatible NadaType.

    Args:
        arr (np.ndarray): Input Numpy array.
        nada_type (_NadaCleartextType): Desired clear-text NadaType.

    Returns:
        NadaArray: Output NadaArray.
    """
    if nada_type == Rational:
        nada_type = rational
    else:
        arr = arr.astype(int)
    return NadaArray(np.frompyfunc(nada_type, 1, 1)(arr))


@copy_metadata(np.pad)
def pad(
    arr: NadaArray,
    pad_width: Union[Iterable[int], int],
    mode: str = "constant",
    **kwargs,
) -> NadaArray:
    if mode not in {"constant", "edge", "reflect", "symmetric", "wrap"}:
        raise NotImplementedError(
            "Not currently possible to pad NadaArray in mode `%s`" % mode
        )

    # Override python defaults by NadaType defaults
    overriden_kwargs = {}
    if mode == "constant":
        dtype = arr.dtype
        if dtype in (Rational, SecretRational):
            nada_type = rational
        elif dtype in (PublicInteger, SecretInteger):
            nada_type = Integer
        elif dtype == (PublicUnsignedInteger, SecretUnsignedInteger):
            nada_type = UnsignedInteger
        else:
            nada_type = dtype

        overriden_kwargs["constant_values"] = kwargs.get(
            "constant_values", nada_type(0)
        )

    padded_inner = np.pad(
        arr,
        pad_width,
        mode,
        **overriden_kwargs,
        **kwargs,
    )

    return NadaArray(padded_inner)


@copy_metadata(np.eye)
def eye(*args, nada_type: _NadaCleartextType, **kwargs) -> NadaArray:
    return to_nada(np.eye(*args, **kwargs), nada_type)


@copy_metadata(np.arange)
def arange(*args, nada_type: _NadaCleartextType, **kwargs) -> NadaArray:
    return to_nada(np.arange(*args, **kwargs), nada_type)


@copy_metadata(np.linspace)
def linspace(*args, nada_type: _NadaCleartextType, **kwargs) -> NadaArray:
    return to_nada(np.linspace(*args, **kwargs), nada_type)


@copy_metadata(np.split)
def split(a: NadaArray, *args, **kwargs) -> NadaArray:
    return NadaArray(np.split(a.inner, *args, **kwargs))


@copy_metadata(np.compress)
def compress(a: NadaArray, *args, **kwargs):
    return a.compress(*args, **kwargs)


@copy_metadata(np.copy)
def copy(a: NadaArray, *args, **kwargs):
    return a.copy(*args, **kwargs)


@copy_metadata(np.cumprod)
def cumprod(a: NadaArray, *args, **kwargs):
    return a.cumprod(*args, **kwargs)


@copy_metadata(np.cumsum)
def cumsum(a: NadaArray, *args, **kwargs):
    return a.cumsum(*args, **kwargs)


@copy_metadata(np.diagonal)
def diagonal(a: NadaArray, *args, **kwargs):
    return a.diagonal(*args, **kwargs)


@copy_metadata(np.prod)
def prod(a: NadaArray, *args, **kwargs):
    return a.prod(*args, **kwargs)


@copy_metadata(np.put)
def put(a: NadaArray, *args, **kwargs):
    return a.put(*args, **kwargs)


@copy_metadata(np.ravel)
def ravel(a: NadaArray, *args, **kwargs):
    return a.ravel(*args, **kwargs)


@copy_metadata(np.repeat)
def repeat(a: NadaArray, *args, **kwargs):
    return a.repeat(*args, **kwargs)


@copy_metadata(np.reshape)
def reshape(a: NadaArray, *args, **kwargs):
    return a.reshape(*args, **kwargs)


@copy_metadata(np.resize)
def resize(a: NadaArray, *args, **kwargs):
    return a.resize(*args, **kwargs)


@copy_metadata(np.squeeze)
def squeeze(a: NadaArray, *args, **kwargs):
    return a.squeeze(*args, **kwargs)


@copy_metadata(np.sum)
def sum(a: NadaArray, *args, **kwargs):
    return a.sum(*args, **kwargs)


@copy_metadata(np.swapaxes)
def swapaxes(a: NadaArray, *args, **kwargs):
    return a.swapaxes(*args, **kwargs)


@copy_metadata(np.take)
def take(a: NadaArray, *args, **kwargs):
    return a.take(*args, **kwargs)


@copy_metadata(np.trace)
def trace(a: NadaArray, *args, **kwargs):
    return a.trace(*args, **kwargs)


@copy_metadata(np.transpose)
def transpose(a: NadaArray, *args, **kwargs):
    return a.transpose(*args, **kwargs)

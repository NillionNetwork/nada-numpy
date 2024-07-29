"""
This module provides common functions to work with Nada Numpy, including the creation
and manipulation of arrays and party objects.
"""

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from nada_dsl import (Boolean, Integer, Output, Party, PublicInteger,
                      PublicUnsignedInteger, SecretInteger,
                      SecretUnsignedInteger, UnsignedInteger)

from nada_numpy.array import NadaArray
from nada_numpy.nada_typing import AnyNadaType, NadaCleartextNumber
from nada_numpy.types import Rational, SecretRational, rational
from nada_numpy.utils import copy_metadata

__all__ = [
    "parties",
    "from_list",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
    "alphas",
    "alphas_like",
    "array",
    "random",
    "output",
    "vstack",
    "hstack",
    "ndim",
    "shape",
    "size",
    "pad",
    "frompyfunc",
    "vectorize",
    "eye",
    "arange",
    "linspace",
    "split",
    "compress",
    "copy",
    "cumprod",
    "cumsum",
    "diagonal",
    "mean",
    "prod",
    "put",
    "ravel",
    "repeat",
    "reshape",
    "resize",
    "squeeze",
    "sum",
    "swapaxes",
    "take",
    "trace",
    "transpose",
]


def parties(num: int, party_names: Optional[List[str]] = None) -> List[Party]:
    """
    Create a list of Party objects with specified names.

    Args:
        num (int): The number of parties to create.
        party_names (List[str], optional): Party names to use. Defaults to None.

    Raises:
        ValueError: Raised when incorrect number of party names is supplied.

    Returns:
        List[Party]: A list of Party objects.
    """
    if party_names is None:
        party_names = [f"Party{i}" for i in range(num)]

    if len(party_names) != num:
        num_supplied_parties = len(party_names)
        raise ValueError(
            f"Incorrect number of party names. Expected {num}, received {num_supplied_parties}"
        )

    return [Party(name=party_name) for party_name in party_names]


def __from_numpy(arr: np.ndarray, nada_type: NadaCleartextNumber) -> List:
    """
    Recursively convert a n-dimensional NumPy array to a nested list of NadaInteger objects.

    Args:
        arr (np.ndarray): A NumPy array of integers.
        nada_type (type): The type of NadaInteger objects to create.

    Returns:
        List: A nested list of NadaInteger objects.
    """
    if len(arr.shape) == 1:
        if isinstance(nada_type, Rational):
            return [nada_type(elem) for elem in arr]  # type: ignore
        return [nada_type(int(elem)) for elem in arr]  # type: ignore
    return [__from_numpy(arr[i], nada_type) for i in range(arr.shape[0])]


def from_list(
    lst: Union[List, np.ndarray], nada_type: NadaCleartextNumber = Integer
) -> NadaArray:
    """
    Create a cleartext NadaArray from a list of integers.

    Args:
        lst (Union[List, np.ndarray]): A list of integers representing the elements of the array.
        nada_type (type, optional): The type of NadaInteger objects to create. Defaults to Integer.

    Returns:
        NadaArray: The created NadaArray.
    """
    if nada_type == Rational:
        nada_type = rational
    lst_np = np.array(lst)
    return NadaArray(np.array(__from_numpy(lst_np, nada_type)))


def ones(dims: Sequence[int], nada_type: NadaCleartextNumber = Integer) -> NadaArray:
    """
    Create a cleartext NadaArray filled with ones.

    Args:
        dims (Sequence[int]): A list of integers representing the dimensions of the array.
        nada_type (type, optional): The type of NadaInteger objects to create. Defaults to Integer.

    Returns:
        NadaArray: The created NadaArray filled with ones.
    """
    if nada_type == Rational:
        nada_type = rational
    return from_list(np.ones(dims), nada_type)


def ones_like(
    a: np.ndarray | NadaArray, nada_type: NadaCleartextNumber = Integer
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


def zeros(dims: Sequence[int], nada_type: NadaCleartextNumber = Integer) -> NadaArray:
    """
    Create a cleartext NadaArray filled with zeros.

    Args:
        dims (Sequence[int]): A list of integers representing the dimensions of the array.
        nada_type (type, optional): The type of NadaInteger objects to create. Defaults to Integer.

    Returns:
        NadaArray: The created NadaArray filled with zeros.
    """
    if nada_type == Rational:
        nada_type = rational
    return from_list(np.zeros(dims), nada_type)


def zeros_like(
    a: np.ndarray | NadaArray, nada_type: NadaCleartextNumber = Integer
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


def alphas(dims: Sequence[int], alpha: Any) -> NadaArray:
    """
    Create a NadaArray filled with a certain constant value.

    Args:
        dims (Sequence[int]): A list of integers representing the dimensions of the array.
        alpha (Any): Some constant value.

    Returns:
        NadaArray: NadaArray filled with constant value.
    """
    ones_array = np.ones(dims)
    return NadaArray(np.frompyfunc(lambda _: alpha, 1, 1)(ones_array))


def alphas_like(a: np.ndarray | NadaArray, alpha: Any) -> NadaArray:
    """
    Create a NadaArray filled with a certain constant value
    with the same shape and type as a given array.

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
    dims: Sequence[int],
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
        dims (Sequence[int]): A list of integers representing the dimensions of the array.
        party (Party): The party object.
        prefix (str): A prefix for naming the array elements.
        nada_type (type): The type of elements to create.

    Returns:
        NadaArray: The created NadaArray.
    """
    return NadaArray.array(dims, party, prefix, nada_type)


def random(
    dims: Sequence[int],
    nada_type: SecretInteger | SecretUnsignedInteger | SecretRational = SecretInteger,
) -> NadaArray:
    """
    Create a random NadaArray with the specified dimensions.

    Args:
        dims (Sequence[int]): A list of integers representing the dimensions of the array.
        nada_type (type, optional): The type of elements to create. Defaults to SecretInteger.

    Returns:
        NadaArray: A NadaArray with random values of the specified type.
    """
    return NadaArray.random(dims, nada_type)


def output(
    value: Union[NadaArray, AnyNadaType], party: Party, prefix: str
) -> List[Output]:
    """
    Generate a list of Output objects for some provided value.

    Args:
        value (Union[NadaArray, AnyNadaType]): The input NadaArray.
        party (Party): The party object.
        prefix (str): The prefix for naming the Output objects.

    Returns:
        List[Output]: A list of Output objects.
    """
    if isinstance(value, NadaArray):
        # pylint:disable=protected-access
        return NadaArray._output_array(value, party, prefix)
    if isinstance(value, (Rational, SecretRational)):
        value = value.value
    return [Output(value, prefix, party)]


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


def to_nada(arr: np.ndarray, nada_type: NadaCleartextNumber) -> NadaArray:
    """
    Converts a plain-text NumPy array to the equivalent NadaArray with
    a specified compatible NadaType.

    Args:
        arr (np.ndarray): Input Numpy array.
        nada_type (NadaCleartextNumber): Desired clear-text NadaType.

    Returns:
        NadaArray: Output NadaArray.
    """
    if nada_type == Rational:
        nada_type = rational
    else:
        arr = arr.astype(int)
    return NadaArray(np.frompyfunc(nada_type, 1, 1)(arr))  # type: ignore


# pylint:disable=missing-function-docstring
@copy_metadata(np.pad)
def pad(
    arr: NadaArray,
    pad_width: Union[Sequence[int], int],
    mode: str = "constant",
    **kwargs,
) -> NadaArray:
    if mode not in {"constant", "edge", "reflect", "symmetric", "wrap"}:
        raise NotImplementedError(
            f"Not currently possible to pad NadaArray in mode `{mode}`"
        )

    # Override python defaults by NadaType defaults
    overriden_kwargs = {}
    if mode == "constant" and "constant_values" not in kwargs:
        if arr.is_rational:
            default = rational(0)
        elif arr.is_integer:
            default = Integer(0)
        elif arr.is_unsigned_integer:
            default = UnsignedInteger(0)
        else:
            default = Boolean(False)

        overriden_kwargs["constant_values"] = kwargs.get("constant_values", default)

    padded_inner = np.pad(  # type: ignore
        arr.inner,
        pad_width,
        mode,
        **overriden_kwargs,
        **kwargs,
    )

    return NadaArray(padded_inner)


# pylint:disable=too-few-public-methods
class NadaCallable:
    """Class that wraps a vectorized NumPy function"""

    def __init__(self, vfunc: Callable) -> None:
        """
        Initialization.

        Args:
            vfunc (Callable): Vectorized function to wrap.
        """
        self.vfunc = vfunc

    def __call__(self, *args, **kwargs) -> Any:
        """
        Routes function call to wrapped vectorized function while
        ensuring any resulting NumPy arrays are converted to NadaArrays.

        Returns:
            Any: Function result.
        """
        result = self.vfunc(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        if isinstance(result, Sequence):
            return type(result)(  # type: ignore
                NadaArray(value) if isinstance(value, np.ndarray) else value
                for value in result
            )
        return result


# pylint:disable=missing-function-docstring
@copy_metadata(np.frompyfunc)
def frompyfunc(*args, **kwargs) -> NadaCallable:
    return NadaCallable(np.frompyfunc(*args, **kwargs))


# pylint:disable=missing-function-docstring
@copy_metadata(np.vectorize)
def vectorize(*args, **kwargs) -> NadaCallable:
    return NadaCallable(np.vectorize(*args, **kwargs))


# pylint:disable=missing-function-docstring
@copy_metadata(np.eye)
def eye(*args, nada_type: NadaCleartextNumber, **kwargs) -> NadaArray:
    return to_nada(np.eye(*args, **kwargs), nada_type)


# pylint:disable=missing-function-docstring
@copy_metadata(np.arange)
def arange(*args, nada_type: NadaCleartextNumber, **kwargs) -> NadaArray:
    return to_nada(np.arange(*args, **kwargs), nada_type)


# pylint:disable=missing-function-docstring
@copy_metadata(np.linspace)
def linspace(*args, nada_type: NadaCleartextNumber, **kwargs) -> NadaArray:
    return to_nada(np.linspace(*args, **kwargs), nada_type)


# pylint:disable=missing-function-docstring
@copy_metadata(np.split)
def split(a: NadaArray, *args, **kwargs) -> List[NadaArray]:
    return [NadaArray(arr) for arr in np.split(a.inner, *args, **kwargs)]


# pylint:disable=missing-function-docstring
@copy_metadata(np.compress)
def compress(a: NadaArray, *args, **kwargs):
    return a.compress(*args, **kwargs)


# pylint:disable=missing-function-docstring
@copy_metadata(np.copy)
def copy(a: NadaArray, *args, **kwargs):
    return a.copy(*args, **kwargs)


# pylint:disable=missing-function-docstring
@copy_metadata(np.cumprod)
def cumprod(a: NadaArray, *args, **kwargs):
    return a.cumprod(*args, **kwargs)


# pylint:disable=missing-function-docstring
@copy_metadata(np.cumsum)
def cumsum(a: NadaArray, *args, **kwargs):
    return a.cumsum(*args, **kwargs)


# pylint:disable=missing-function-docstring
@copy_metadata(np.diagonal)
def diagonal(a: NadaArray, *args, **kwargs):
    return a.diagonal(*args, **kwargs)


# pylint:disable=missing-function-docstring
@copy_metadata(np.diagonal)
def mean(a: NadaArray, *args, **kwargs):
    return a.mean(*args, **kwargs)


# pylint:disable=missing-function-docstring
@copy_metadata(np.prod)
def prod(a: NadaArray, *args, **kwargs):
    return a.prod(*args, **kwargs)


# pylint:disable=missing-function-docstring
@copy_metadata(np.put)
def put(a: NadaArray, *args, **kwargs):
    return a.put(*args, **kwargs)


# pylint:disable=missing-function-docstring
@copy_metadata(np.ravel)
def ravel(a: NadaArray, *args, **kwargs):
    return a.ravel(*args, **kwargs)


# pylint:disable=missing-function-docstring
@copy_metadata(np.repeat)
def repeat(a: NadaArray, *args, **kwargs):
    return a.repeat(*args, **kwargs)


# pylint:disable=missing-function-docstring
@copy_metadata(np.reshape)
def reshape(a: NadaArray, *args, **kwargs):
    return a.reshape(*args, **kwargs)


# pylint:disable=missing-function-docstring
@copy_metadata(np.resize)
def resize(a: NadaArray, *args, **kwargs):
    return a.resize(*args, **kwargs)


# pylint:disable=missing-function-docstring
@copy_metadata(np.squeeze)
def squeeze(a: NadaArray, *args, **kwargs):
    return a.squeeze(*args, **kwargs)


# pylint:disable=missing-function-docstring,redefined-builtin
@copy_metadata(np.sum)
def sum(a: NadaArray, *args, **kwargs):
    return a.sum(*args, **kwargs)


# pylint:disable=missing-function-docstring
@copy_metadata(np.swapaxes)
def swapaxes(a: NadaArray, *args, **kwargs):
    return a.swapaxes(*args, **kwargs)


# pylint:disable=missing-function-docstring
@copy_metadata(np.take)
def take(a: NadaArray, *args, **kwargs):
    return a.take(*args, **kwargs)


# pylint:disable=missing-function-docstring
@copy_metadata(np.trace)
def trace(a: NadaArray, *args, **kwargs):
    return a.trace(*args, **kwargs)


# pylint:disable=missing-function-docstring
@copy_metadata(np.transpose)
def transpose(a: NadaArray, *args, **kwargs):
    return a.transpose(*args, **kwargs)

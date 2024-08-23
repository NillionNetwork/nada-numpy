"""
This module provides common functions to work with Nada Numpy, including the creation
and manipulation of arrays and party objects.
"""

# pylint:disable=too-many-lines

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
    "sign",
    "abs",
    "exp",
    "polynomial",
    "log",
    "reciprocal",
    "inv_sqrt",
    "sqrt",
    "cossin",
    "sin",
    "cos",
    "tan",
    "tanh",
    "sigmoid",
    "gelu",
    "silu",
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


# Non-linear functions


def sign(arr: NadaArray) -> "NadaArray":
    """Computes the sign value (0 is considered positive)"""
    return arr.sign()


def abs(arr: NadaArray) -> "NadaArray":
    """Computes the absolute value"""
    return arr.abs()


def exp(arr: NadaArray, iterations: int = 8) -> "NadaArray":
    """
    Approximates the exponential function using a limit approximation.

    The exponential function is approximated using the following limit:

        exp(x) = lim_{n -> ∞} (1 + x / n) ^ n

    The exponential function is computed by choosing n = 2 ** d, where d is set to `iterations`.
    The calculation is performed by computing (1 + x / n) once and then squaring it `d` times.

    Approximation accuracy range (with 16 bit precision):
    + ---------------------------------- +
    |  Input range x |  Relative error   |
    + ---------------------------------- +
    |   [-2, 2]      |       <1%         |
    |   [-7, 7]      |       <10%        |
    |   [-8, 15]     |       <35%        |
    + ---------------------------------- +

    Args:
        iterations (int, optional): The number of iterations for the limit approximation.
            Defaults to 8.

    Returns:
        NadaArray: The approximated value of the exponential function.
    """
    return arr.exp(iterations=iterations)


def polynomial(arr: NadaArray, coefficients: list) -> "NadaArray":
    """
    Computes a polynomial function on a value with given coefficients.

    The coefficients can be provided as a list of values.
    They should be ordered from the linear term (order 1) first, ending with the highest order term.
    **Note: The constant term is not included.**

    Args:
        coefficients (list): The coefficients of the polynomial, ordered by increasing degree.

    Returns:
        NadaArray: The result of the polynomial function applied to the input x.
    """
    return arr.polynomial(coefficients=coefficients)


def log(
    arr: NadaArray,
    input_in_01: bool = False,
    iterations: int = 2,
    exp_iterations: int = 8,
    order: int = 8,
) -> "NadaArray":
    """
    Approximates the natural logarithm using 8th order modified Householder iterations.
    This approximation is accurate within 2% relative error on the interval [0.0001, 250].

    The iterations are computed as follows:

        h = 1 - x * exp(-y_n)
        y_{n+1} = y_n - sum(h^k / k for k in range(1, order + 1))

    Approximation accuracy range (with 16 bit precision):
    + ------------------------------------- +
    |    Input range x  |  Relative error   |
    + ------------------------------------- +
    | [0.001, 200]      |     <1%           |
    | [0.00003, 253]    |     <10%          |
    | [0.0000001, 253]  |     <40%          |
    | [253, +∞[         |     Unstable      |
    + ------------------------------------- +

    Args:
        input_in_01 (bool, optional): Indicates if the input is within the domain [0, 1].
            This setting optimizes the function for this domain, useful for computing
            log-probabilities in entropy functions.

            To shift the domain of convergence, a constant 'a' is used with the identity:

                ln(u) = ln(au) - ln(a)

            Given the convergence domain for log() function is approximately [1e-4, 1e2],
            we set a = 100.
            Defaults to False.
        iterations (int, optional): Number of Householder iterations for the approximation.
            Defaults to 2.
        exp_iterations (int, optional): Number of iterations for the limit approximation of exp.
            Defaults to 8.
        order (int, optional): Number of polynomial terms used (order of Householder approximation).
            Defaults to 8.

    Returns:
        NadaArray: The approximate value of the natural logarithm.
    """
    return arr.log(
        input_in_01=input_in_01,
        iterations=iterations,
        exp_iterations=exp_iterations,
        order=order,
    )


def reciprocal(  # pylint: disable=too-many-arguments
    arr: NadaArray,
    all_pos: bool = False,
    initial: Optional["Rational"] = None,
    input_in_01: bool = False,
    iterations: int = 10,
    log_iters: int = 1,
    exp_iters: int = 8,
    method: str = "NR",
) -> "NadaArray":
    r"""
    Approximates the reciprocal of a number through two possible methods: Newton-Raphson
    and log.

    Methods:
        'NR' : `Newton-Raphson`_ method computes the reciprocal using iterations
                of :math:`x_{i+1} = (2x_i - x * x_i^2)` and uses
                :math:`3*exp(1 - 2x) + 0.003` as an initial guess by default.

                Approximation accuracy range (with 16 bit precision):
                + ------------------------------------ +
                | Input range |x|  |  Relative error   |
                + ------------------------------------ +
                | [0.1, 64]        |       <0%         |
                | [0.0003, 253]    |       <15%        |
                | [0.00001, 253]   |       <90%        |
                | [253, +∞[        |     Unstable      |
                + ------------------------------------ +

        'log' : Computes the reciprocal of the input from the observation that:
                :math:`x^{-1} = exp(-log(x))`

                Approximation accuracy range (with 16 bit precision):
                + ------------------------------------ +
                | Input range |x|  |  Relative error   |
                + ------------------------------------ +
                | [0.0003, 253]    |       <15%        |
                | [0.00001, 253]   |       <90%        |
                | [253, +∞[        |     Unstable      |
                + ------------------------------------ +

        Args:
            all_pos (bool, optional): determines whether all elements of the
                input are known to be positive, which optimizes the step of
                computing the sign of the input. Defaults to False.
            initial (Rational, optional): sets the initial value for the
                Newton-Raphson method. By default, this will be set to :math:
                `3*exp(-(x-.5)) + 0.003` as this allows the method to converge over
                a fairly large domain.
            input_in_01 (bool, optional) : Allows a user to indicate that the input is
                        in the range [0, 1], causing the function optimize for this range.
                        This is useful for improving the accuracy of functions on
                        probabilities (e.g. entropy functions).
            iterations (int, optional):  determines the number of Newton-Raphson iterations to run
                            for the `NR` method. Defaults to 10.
            log_iters (int, optional): determines the number of Householder
                iterations to run when computing logarithms for the `log` method. Defaults to 1.
            exp_iters (int, optional): determines the number of exp
                iterations to run when computing exp. Defaults to 8.
            method (str, optional): method used to compute reciprocal. Defaults to "NR".

    Returns:
        NadaArray: The approximate value of the reciprocal

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Newton%27s_method
    """
    # pylint:disable=duplicate-code
    return arr.reciprocal(
        all_pos=all_pos,
        initial=initial,
        input_in_01=input_in_01,
        iterations=iterations,
        log_iters=log_iters,
        exp_iters=exp_iters,
        method=method,
    )


def inv_sqrt(
    arr: NadaArray,
    initial: Optional["SecretRational"] = None,
    iterations: int = 5,
    method: str = "NR",
) -> "NadaArray":
    r"""
    Computes the inverse square root of the input using the Newton-Raphson method.

    Approximation accuracy range (with 16 bit precision):
    + ---------------------------------- +
    | Input range x  |  Relative error   |
    + ---------------------------------- +
    | [0.1, 170]     |       <0%         |
    | [0.001, 200]   |       <50%        |
    | [200, +∞[      |     Unstable      |
    + ---------------------------------- +

    Args:
        initial (Union[SecretRational, None], optional): sets the initial value for the
                    Newton-Raphson iterations. By default, this will be set to allow the
                    method to converge over a fairly large domain.
        iterations (int, optional): determines the number of Newton-Raphson iterations to run.
        method (str, optional): method used to compute inv_sqrt. Defaults to "NR".

    Returns:
        NadaArray: The approximate value of the inv_sqrt.

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """
    return arr.inv_sqrt(initial=initial, iterations=iterations, method=method)


def sqrt(
    arr: NadaArray,
    initial: Optional["SecretRational"] = None,
    iterations: int = 5,
    method: str = "NR",
) -> "NadaArray":
    r"""
    Computes the square root of the input by computing its inverse square root using
    the Newton-Raphson method and multiplying by the input.

    Approximation accuracy range (with 16 bit precision):
    + ---------------------------------- +
    | Input range x  |  Relative error   |
    + ---------------------------------- +
    | [0.1, 170]     |       <0%         |
    | [0.001, 200]   |       <50%        |
    | [200, +∞[      |     Unstable      |
    + ---------------------------------- +

    Args:
        initial (Union[SecretRational, None], optional): sets the initial value for the inverse
            square root Newton-Raphson iterations. By default, this will be set to allow
            convergence over a fairly large domain. Defaults to None.
        iterations (int, optional):  determines the number of Newton-Raphson iterations to run.
            Defaults to 5.
        method (str, optional): method used to compute sqrt. Defaults to "NR".

    Returns:
        NadaArray: The approximate value of the sqrt.

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """
    return arr.sqrt(initial=initial, iterations=iterations, method=method)


# Trigonometry


def cossin(arr: NadaArray, iterations: int = 10) -> "NadaArray":
    r"""Computes cosine and sine through e^(i * input) where i is the imaginary unit through the
    formula:

    .. math::
        Re\{e^{i * input}\}, Im\{e^{i * input}\} = \cos(input), \sin(input)

    Args:
        iterations (int, optional): determines the number of iterations to run. Defaults to 10.

    Returns:
        Tuple[NadaArray, NadaArray]:
            A tuple where the first element is cos and the second element is the sin.
    """
    return arr.cossin(iterations=iterations)


def cos(arr: NadaArray, iterations: int = 10) -> "NadaArray":
    r"""Computes the cosine of the input using cos(x) = Re{exp(i * x)}.

    Note: unstable outside [-30, 30]

    Args:
        iterations (int, optional): determines the number of iterations to run. Defaults to 10.

    Returns:
        NadaArray: The approximate value of the cosine.
    """
    return arr.cos(iterations=iterations)


def sin(arr: NadaArray, iterations: int = 10) -> "NadaArray":
    r"""Computes the sine of the input using sin(x) = Im{exp(i * x)}.

    Note: unstable outside [-30, 30]

    Args:
        iterations (int, optional): determines the number of iterations to run. Defaults to 10.

    Returns:
        NadaArray: The approximate value of the sine.
    """
    return arr.sin(iterations=iterations)


def tan(arr: NadaArray, iterations: int = 10) -> "NadaArray":
    r"""Computes the tan of the input using tan(x) = sin(x) / cos(x).

    Note: unstable outside [-30, 30]

    Args:
        iterations (int, optional): determines the number of iterations to run. Defaults to 10.

    Returns:
        NadaArray: The approximate value of the tan.
    """
    return arr.tan(iterations=iterations)


# Activation functions


def tanh(
    arr: NadaArray,
    chebyshev_terms: int = 32,
    method: str = "reciprocal",
) -> "NadaArray":
    r"""Computes the hyperbolic tangent function using the identity

    .. math::
        tanh(x) = 2\sigma(2x) - 1

    Methods:
    If a valid method is given, this function will compute tanh using that method:

        "reciprocal" - computes tanh using the identity

            .. math::
            tanh(x) = 2\sigma(2x) - 1

            Note: stable for x in [-250, 250]. Unstable otherwise.

        "chebyshev" - computes tanh via Chebyshev approximation with truncation.

            .. math::
                tanh(x) = \sum_{j=1}^chebyshev_terms c_{2j - 1} P_{2j - 1} (x / maxval)

            where c_i is the ith Chebyshev series coefficient and P_i is ith polynomial.

            Note: stable for all input range as the approximation is truncated
                    to +/-1 outside [-1, 1].

        "motzkin" - computes tanh via approximation from the paper
            "BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers"
            on section 5.3 based on the Motzkin’s polynomial preprocessing technique.

            Note: stable for all input range as the approximation is truncated
                    to +/-1 outside [-1, 1].

    Args:
        chebyshev_terms (int, optional): highest degree of Chebyshev polynomials.
                        Must be even and at least 6. Defaults to 32.
        method (str, optional): method used to compute tanh function. Defaults to "reciprocal".

    Returns:
        NadaArray: The tanh evaluation.

    Raises:
        ValueError: Raised if method type is not supported.
    """
    return arr.tanh(chebyshev_terms=chebyshev_terms, method=method)


def sigmoid(
    arr: NadaArray,
    chebyshev_terms: int = 32,
    method: str = "reciprocal",
) -> "NadaArray":
    r"""Computes the sigmoid function using the following definition

    .. math::
        \sigma(x) = (1 + e^{-x})^{-1}

    Methods:
    If a valid method is given, this function will compute sigmoid
        using that method:

        "chebyshev" - computes tanh via Chebyshev approximation with
            truncation and uses the identity:

            .. math::
                \sigma(x) = \frac{1}{2}tanh(\frac{x}{2}) + \frac{1}{2}

            Note: stable for all input range as the approximation is truncated
                    to 0/1 outside [-1, 1].

        "motzkin" - computes tanh via approximation from the paper
            "BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers"
            on section 5.3 based on the Motzkin’s polynomial preprocessing technique. It uses
            the identity:

            .. math::
                \sigma(x) = \frac{1}{2}tanh(\frac{x}{2}) + \frac{1}{2}

            Note: stable for all input range as the approximation is truncated
                    to 0/1 outside [-1, 1].

        "reciprocal" - computes sigmoid using :math:`1 + e^{-x}` and computing
            the reciprocal

            Note: stable for x in [-500, 500]. Unstable otherwise.

    Args:
        chebyshev_terms (int, optional): highest degree of Chebyshev polynomials.
                        Must be even and at least 6. Defaults to 32.
        method (str, optional): method used to compute sigmoid function. Defaults to "reciprocal".

    Returns:
        NadaArray: The sigmoid evaluation.

    Raises:
        ValueError: Raised if method type is not supported.
    """

    return arr.sigmoid(chebyshev_terms=chebyshev_terms, method=method)


def gelu(
    arr: NadaArray,
    method: str = "tanh",
    tanh_method: str = "reciprocal",
) -> "NadaArray":
    r"""Computes the gelu function using the following definition

    .. math::
        gelu(x) = x/2 * (1 + tanh(\sqrt{2/\pi} * (x + 0.04471 * x^3)))

    Methods:
    If a valid method is given, this function will compute gelu
        using that method:

        "tanh" - computes gelu using the common approximation function

            Note: stable for x in [-18, 18]. Unstable otherwise.

        "motzkin" - computes gelu via approximation from the paper
            "BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers"
            on section 5.2 based on the Motzkin’s polynomial preprocessing technique.

            Note: stable for all input range as the approximation is truncated
            to relu function outside [-2.7, 2.7].

    Args:
        method (str, optional): method used to compute gelu function. Defaults to "tanh".
        tanh_method (str, optional): method used for tanh function. Defaults to "reciprocal".

    Returns:
        NadaArray: The gelu evaluation.

    Raises:
        ValueError: Raised if method type is not supported.
    """

    return arr.gelu(method=method, tanh_method=tanh_method)


def silu(
    arr: NadaArray,
    method_sigmoid: str = "reciprocal",
) -> "NadaArray":
    r"""Computes the gelu function using the following definition

    .. math::
        silu(x) = x * sigmoid(x)

    Sigmoid methods:
    If a valid method is given, this function will compute sigmoid
        using that method:

        "chebyshev" - computes tanh via Chebyshev approximation with
            truncation and uses the identity:

            .. math::
                \sigma(x) = \frac{1}{2}tanh(\frac{x}{2}) + \frac{1}{2}

            Note: stable for all input range as the approximation is truncated
                    to 0/1 outside [-1, 1].

        "motzkin" - computes tanh via approximation from the paper
            "BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers"
            on section 5.3 based on the Motzkin’s polynomial preprocessing technique. It uses the
            identity:

            .. math::
                \sigma(x) = \frac{1}{2}tanh(\frac{x}{2}) + \frac{1}{2}

            Note: stable for all input range as the approximation is truncated
                    to 0/1 outside [-1, 1].

        "reciprocal" - computes sigmoid using :math:`1 + e^{-x}` and computing
            the reciprocal

            Note: stable for x in [-500, 500]. Unstable otherwise.

    Args:
        method_sigmoid (str, optional): method used to compute sigmoid function.
            Defaults to "reciprocal".

    Returns:
        NadaArray: The sigmoid evaluation.

    Raises:
        ValueError: Raised if sigmoid method type is not supported.
    """
    return arr.silu(method_sigmoid=method_sigmoid)

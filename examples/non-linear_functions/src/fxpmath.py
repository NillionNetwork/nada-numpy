# Fixed-point math operations

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Part of the code is from the CrypTen Facebook Project:
# https://github.com/facebookresearch/CrypTen/blob/main/crypten/common/functions/logic.py
# https://github.com/facebookresearch/CrypTen/blob/main/crypten/common/functions/approximations.py
#
# Modifications:
# July, 2024
#   - Nada datatypes.
#   - Relative accuracy documentation. 
#   - Some performance improvements.
#   - Fixed Tanh Chebyshev method by changing '_hardtanh' implementation.
#   - Tan.
#   - Motzkin's prolynomial preprocessing approach.
#   - GeLU and SiLU functions.

from typing import Union, Tuple, TypeVar
import functools
import numpy as np

from nada_dsl import UnsignedInteger
# from nada_numpy import (NadaArray, Rational, SecretRational, rational)
from nada_numpy import (NadaArray, rational)

# _NadaRational = Union["Rational", "SecretRational"]
_NadaRational = TypeVar('_NadaRational')

__all__ = [
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
    "silu"
]

def sign(input: _NadaRational) -> _NadaRational:
    """Computes the sign value (0 is considered positive)"""
    
    ltz_cond = input < rational(0)
    ltz = ltz_cond.if_else(rational(1), rational(0))
    
    return rational(1) - rational(2) * ltz

def abs(input: _NadaRational) -> _NadaRational:
    """Computes the absolute value"""
    return input * sign(input)

def exp(
        x: _NadaRational, 
        iterations: int = 8
    ) -> _NadaRational:
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
        x (Union[Rational, SecretRational]): The exponent for which the exponential function is to be approximated.
        iterations (int, optional): The number of iterations for the limit approximation. Defaults to 8.

    Returns:
        Union[Rational, SecretRational]: The approximated value of the exponential function.
    """
    iters_na = UnsignedInteger(iterations)

    result = rational(1) + (x >> iters_na)
    for _ in range(iterations):
        result = result ** 2
    return result



def polynomial(
        x: _NadaRational, 
        coefficients: list
    ) -> _NadaRational:
    """
    Computes a polynomial function on a value with given coefficients.

    The coefficients can be provided as a list of values.
    They should be ordered from the linear term (order 1) first, ending with the highest order term.
    **Note: The constant term is not included.**

    Args:
        x (Union[Rational, SecretRational]): Input value.
        coefficients (list): The coefficients of the polynomial, ordered by increasing degree.

    Returns:
        Union[Rational, SecretRational]: The result of the polynomial function applied to the input x.
    """
    result = rational(0)
    
    for power, coeff in enumerate(coefficients, start=1):
        result += coeff * (x ** power)
    
    return result



def log(
        x: _NadaRational,
        input_in_01: bool = False, 
        iterations: int = 2, 
        exp_iterations: int = 8, 
        order: int = 8
    ) -> _NadaRational:
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
        x (Union[Rational, SecretRational]): The value for which the log function is to be approximated.
        input_in_01 (bool, optional): Indicates if the input is within the domain [0, 1].
            This setting optimizes the function for this domain, useful for computing
            log-probabilities in entropy functions.

            To shift the domain of convergence, a constant 'a' is used with the identity:

                ln(u) = ln(au) - ln(a)

            Given the convergence domain for log() function is approximately [1e-4, 1e2],
            we set a = 100.
            Defaults to False.
        iterations (int, optional): Number of Householder iterations for the approximation. Defaults to 2.
        exp_iterations (int, optional): Number of iterations for the limit approximation of exp. Defaults to 8.
        order (int, optional): Number of polynomial terms used (order of Householder approximation). Defaults to 8.

    Returns:
        Union[Rational, SecretRational]: The approximate value of the natural logarithm.
    """
    if input_in_01:
        return log(x * rational(100), iterations=iterations, exp_iterations=exp_iterations, order=order) - rational(4.605170)

    # Initialization to a decent estimate (found by qualitative inspection):
    #                ln(x) = x/120 - 20exp(-2x - 1.0) + 3.0
    term1 = x * rational(1/120.0)
    term2 = exp( - x - x - rational(1), iterations=exp_iterations) * rational(20)
    y = term1 - term2 + rational(3.0)

    # 8th order Householder iterations
    for _ in range(iterations):
        h = rational(1) - x * exp(-y, iterations=exp_iterations)
        y -= polynomial(h, [rational(1 / (i + 1)) for i in range(order)])
    return y

def reciprocal(
        x: _NadaRational, 
        all_pos: bool = False, 
        initial: Union[_NadaRational, None] = None, 
        input_in_01: bool = False, 
        iterations: int = 10, 
        log_iters: int = 1, 
        exp_iters: int = 8, 
        method: str = "NR"
    ) -> _NadaRational:
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
        x (Union[Rational, SecretRational]): the value for which the reciprocal function is to be approximated.
        all_pos (bool, optional): determines whether all elements of the
            input are known to be positive, which optimizes the step of
            computing the sign of the input. Defaults to False.
        initial (Union[Rational, SecretRational, None], optional): sets the initial value for the
            Newton-Raphson method. By default, this will be set to :math:
            `3*exp(-(x-.5)) + 0.003` as this allows the method to converge over
            a fairly large domain.
        input_in_01 (bool, optional) : Allows a user to indicate that the input is in the range [0, 1],
                    causing the function optimize for this range. This is useful for improving
                    the accuracy of functions on probabilities (e.g. entropy functions).
        iterations (int, optional):  determines the number of Newton-Raphson iterations to run
                        for the `NR` method. Defaults to 10.
        log_iters (int): determines the number of Householder
            iterations to run when computing logarithms for the `log` method. Defaults to 1.
        exp_iters (int): determines the number of exp
            iterations to run when computing exp. Defaults to 8.
        method (str, optional): method used to compute reciprocal. Defaults to "NR".

    Returns:
        Union[Rational, SecretRational]: The approximate value of the reciprocal

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Newton%27s_method
    """
    if input_in_01:
        rec = reciprocal(x * rational(64), method=method, \
                         all_pos=True, initial=initial, \
                            iterations=iterations) * rational(64)
        return rec

    if not all_pos:
        sgn = sign(x)
        pos = sgn * x
        return sgn * reciprocal(pos, method=method, all_pos=True, \
                                initial=initial, iterations=iterations)

    if method == "NR":
        if initial is None:
            # Initialization to a decent estimate (found by qualitative inspection):
            #                1/x = 3exp(1 - 2x) + 0.003
            result = rational(3) * exp(rational(1) - x - x, iterations=exp_iters) \
                + rational(0.003)
        else:
            result = initial
        for _ in range(iterations):
            result = result + result - result * result * x
        return result
    elif method == "log":
        return exp(-log(x, iterations=log_iters), iterations=exp_iters)
    else:
        raise ValueError(f"Invalid method {method} given for reciprocal function")
    
def inv_sqrt(
        x: _NadaRational, 
        initial: Union[_NadaRational, None] = None, 
        iterations: int = 5, 
        method: str = "NR"
    ) -> _NadaRational:
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
        x (Union[Rational, SecretRational]): the value for which the inv_sqrt function is to be approximated.
        initial (Union[Rational, SecretRational, None], optional): sets the initial value for the Newton-Raphson iterations.
                    By default, this will be set to allow the method to converge over a
                    fairly large domain.
        iterations (int, optional): determines the number of Newton-Raphson iterations to run.
        method (str, optional): method used to compute inv_sqrt. Defaults to "NR".

    Returns:
        Union[Rational, SecretRational]: The approximate value of the inv_sqrt.

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """

    if method == "NR":
        if initial is None:
            # Initialization to a decent estimate (found by qualitative inspection):
            #                 exp(- x/2 - 0.2) * 2.2 + 0.2 - x/1024
            y = exp( - (x >> UnsignedInteger(1)) - rational(0.2)) * \
                rational(2.2) + rational(0.2)
            y -= x >> UnsignedInteger(10) # div by 1024
        else:
            y = initial

        # Newton Raphson iterations for inverse square root
        for _ in range(iterations):
            y = (y * (rational(3) - x * y * y)) >> UnsignedInteger(1)
        return y
    else:
        raise ValueError(f"Invalid method {method} given for inv_sqrt function")
    

def sqrt(
        x: _NadaRational, 
        initial: Union[_NadaRational, None] = None,  
        iterations: int = 5,
        method: str = "NR", 
    ) -> _NadaRational:
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
        x (Union[Rational, SecretRational]): the value for which the sqrt function is to be approximated.
        initial (Union[Rational, SecretRational, None], optional): sets the initial value for the inverse square root
            Newton-Raphson iterations. By default, this will be set to allow convergence
            over a fairly large domain. Defaults to None.
        iterations (int, optional):  determines the number of Newton-Raphson iterations to run. Defaults to 5.
        method (str, optional): method used to compute sqrt. Defaults to "NR".

    Returns:
        Union[Rational, SecretRational]: The approximate value of the sqrt.

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """

    if method == "NR":
        return inv_sqrt(x, initial=initial, iterations=iterations, method=method) * x
    else:
        raise ValueError(f"Invalid method {method} given for sqrt function")

# Trigonometry    

def _eix(
        input: _NadaRational, 
        iterations: int = 10
    ) -> Tuple[_NadaRational, _NadaRational]:
    r"""Computes e^(i * input) where i is the imaginary unit through the formula:

    .. math::
        Re\{e^{i * input}\}, Im\{e^{i * input}\} = \cos(input), \sin(input)

    Args:
        intput (Union[Rational, SecretRational]): the input value.
        iterations (int, optional): determines the number of iterations to run. Defaults to 10.

    Returns:
        Tuple[Union[Rational, SecretRational], Union[Rational, SecretRational]]: 
            A tuple where the first element is cos and the second element is the sin.
    """

    re = rational(1)
    im = input >> UnsignedInteger(iterations)

    # First iteration uses knowledge that `re` is public and = 1
    re -= im * im
    im *= rational(2)

    # Compute (a + bi)^2 -> (a^2 - b^2) + (2ab)i `iterations` times
    for _ in range(iterations - 1):
        a2 = re * re
        b2 = im * im
        im = im * re
        im *= rational(2)
        re = a2 - b2

    return re, im

def cossin(
        input: _NadaRational, 
        iterations: int = 10
    ) -> Tuple[_NadaRational, _NadaRational]:
    r"""Computes cosine and sine through e^(i * input) where i is the imaginary unit through the formula:

    .. math::
        Re\{e^{i * input}\}, Im\{e^{i * input}\} = \cos(input), \sin(input)

    Args:
        intput (Union[Rational, SecretRational]): the input value.
        iterations (int, optional): determines the number of iterations to run. Defaults to 10.

    Returns:
        Tuple[Union[Rational, SecretRational], Union[Rational, SecretRational]]: 
            A tuple where the first element is cos and the second element is the sin.
    """
    return _eix(input, iterations=iterations)
    
def cos(
        input: _NadaRational, 
        iterations: int = 10
    ) -> _NadaRational:
    r"""Computes the cosine of the input using cos(x) = Re{exp(i * x)}.

    Note: unstable outside [-30, 30]

    Args:
        intput (Union[Rational, SecretRational]): the input value.
        iterations (int, optional): determines the number of iterations to run. Defaults to 10.

    Returns:
        Union[Rational, SecretRational]: The approximate value of the cosine.
    """
    return cossin(input, iterations=iterations)[0]


def sin(
        input: _NadaRational, 
        iterations: int = 10
    ) -> _NadaRational:
    r"""Computes the sine of the input using sin(x) = Im{exp(i * x)}.

    Note: unstable outside [-30, 30]

    Args:
        intput (Union[Rational, SecretRational]): the input value.
        iterations (int, optional): determines the number of iterations to run. Defaults to 10.

    Returns:
        Union[Rational, SecretRational]: The approximate value of the sine.
    """
    return cossin(input, iterations=iterations)[1]

def tan(
        input: _NadaRational, 
        iterations: int = 10
    ) -> _NadaRational:
    r"""Computes the tan of the input using tan(x) = sin(x) / cos(x).

    Note: unstable outside [-30, 30]
    
    Args:
        intput (Union[Rational, SecretRational]): the input value.
        iterations (int, optional): determines the number of iterations to run. Defaults to 10.

    Returns:
        Union[Rational, SecretRational]: The approximate value of the tan.
    """
    c, s = cossin(input, iterations=iterations)
    return s * reciprocal(c)

# Activation functions

@functools.lru_cache(maxsize=10)
def chebyshev_series(func, width, terms):
    r"""Computes Chebyshev coefficients

    For n = terms, the ith Chebyshev series coefficient is

    .. math::
        c_i = 2/n \sum_{k=1}^n \cos(j(2k-1)\pi / 4n) f(w\cos((2k-1)\pi / 4n))

    Args:
        func (function): function to be approximated
        width (int): approximation will support inputs in range [-width, width]
        terms (int): number of Chebyshev terms used in approximation

    Returns:
        Chebyshev coefficients with shape equal to num of terms.
    """
    n_range = np.arange(start=0, stop=terms, dtype=float)
    x = width * np.cos((n_range + 0.5) * np.pi / terms)
    y = func(x)
    cos_term = np.cos(np.outer(n_range, n_range + 0.5) * np.pi / terms)
    coeffs = (2 / terms) * np.sum(y * cos_term, axis=1)
    return coeffs

def tanh(
        input: _NadaRational, 
        chebyshev_terms: int = 32,
        method: str = "reciprocal" 
    ) -> _NadaRational:
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
        intput (Union[Rational, SecretRational]): the input value.
        chebyshev_terms (int, optional): highest degree of Chebyshev polynomials.
                        Must be even and at least 6. Defaults to 32.
        method (str, optional): method used to compute tanh function. Defaults to "reciprocal".

    Returns:
        Union[Rational, SecretRational]: The tanh evaluation.

    Raises:
        ValueError: Raised if method type is not supported.
    """

    if method == "reciprocal":
        return sigmoid(input + input, method=method) * rational(2) - rational(1)
    elif method == "chebyshev":
        coeffs = chebyshev_series(np.tanh, 1, chebyshev_terms)[1::2]
        # transform np.array of float to na.array of rationals
        coeffs = NadaArray(np.vectorize(rational)(coeffs))
        tanh_polys = _chebyshev_polynomials(input, chebyshev_terms)
        tanh_polys_flipped = tanh_polys.transpose()
        out = tanh_polys_flipped @ coeffs
        # truncate outside [-maxval, maxval]
        return _hardtanh(input, out)
    elif method == "motzkin":
        # Using approximation from "BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers"
        # section 5.3 based on the Motzkin’s polynomial preprocessing technique.

        # ltz is used for absolute value of input and to compute sign (used to generate result). 
        # We don't use 'abs' and 'sign' functions to avoid computing ltz twice.
        ltz_cond = input < rational(0)
        ltz = ltz_cond.if_else(rational(1), rational(0))
        # sign
        sgn =  rational(1) - rational(2) * ltz
        # absolute value
        abs_input = input * sgn

        # Motzkin’s polynomial preprocessing
        t0 = rational(-4.259314087994767)
        t1 = rational(18.86353816972803)
        t2 = rational(-36.42402897526823)
        t3 = rational(-0.013232131886235352)
        t4 = rational(-3.3289339650097993)
        t5 = rational(-0.0024920889620412097)
        tanh_p0 = (abs_input + t0) * abs_input + t1
        tanh_p1 = (tanh_p0 + abs_input + t2) * tanh_p0 * t3 * abs_input + t4 * abs_input + t5

        cond_2_855 = abs_input > rational(2.855)
        result = cond_2_855.if_else(sgn, sgn * tanh_p1)

        return result
    else:
        raise ValueError(f"Unrecognized method {method} for tanh")

### Auxiliary functions for tanh

def _chebyshev_polynomials(
        input: _NadaRational, 
        terms: int
    ) -> NadaArray:
    r"""Evaluates odd degree Chebyshev polynomials at input.

    Chebyshev Polynomials of the first kind are defined as:

    .. math::
        P_0(x) = 1, P_1(x) = x, P_n(x) = 2 P_{n - 1}(x) - P_{n-2}(x)

    Args:
        input (Union["Rational", "SecretRational"]): input at which polynomials are evaluated
        terms (int): highest degree of Chebyshev polynomials.
                        Must be even and at least 6.
    Returns:
        NadaArray of polynomials evaluated at input of shape `(terms, *input)`.

    Raises:
        ValueError: Raised if 'terrms' is odd and < 6.
    """
    if terms % 2 != 0 or terms < 6:
        raise ValueError("Chebyshev terms must be even and >= 6")

    # Initiate base polynomials
    # P_0
    polynomials = np.array([input])
    y = rational(4) * input * input - rational(2)
    z = y - rational(1)
    # P_1
    polynomials = np.append(polynomials, z * input)

    # Generate remaining Chebyshev polynomials using the recurrence relation
    for k in range(2, terms // 2):
        next_polynomial = y * polynomials[k - 1] - polynomials[k - 2]
        polynomials = np.append(polynomials, next_polynomial)

    return polynomials

def _hardtanh(
        input: _NadaRational, 
        output: _NadaRational, 
        abs_const: _NadaRational = rational(1),
        abs_range: _NadaRational = rational(1)
    ) -> _NadaRational: 
    r"""Applies the HardTanh function element-wise.

    HardTanh is defined as:

    .. math::
        \text{HardTanh}(x) = \begin{cases}
            1 & \text{ if } x > 1 \\
            -1 & \text{ if } x < -1 \\
            Tanh(x) & \text{ otherwise } \\
        \end{cases}

    The range of the linear region :math:`[-1, 1]` can be adjusted using
    :attr:`abs_range`.

    Args:
        input (Union[Rational, SecretRational]): the input value of the Tanh.
        output (Union[Rational, SecretRational]): the output value of the approximation of Tanh.
        abs_const (Union[Rational, SecretRational]): constant value to which |Tanh(x)| converges. Defaults to 1.
        abs_range (Union[Rational, SecretRational]): absolute value of the range. Defaults to 1.

    Returns:
        Union[Rational, SecretRational]: HardTanh output.
    """
    # absolute value
    sgn =  sign(input)
    abs_input = input * sgn
    # chekc if inside [-abs_range, abs_range] interval
    ineight_cond = abs_input < abs_range
    result = ineight_cond.if_else(output, abs_const * sgn) 

    return result

### End of auxiliary functions for tanh

def sigmoid(
        input: _NadaRational, 
        chebyshev_terms: int = 32,
        method: str = "reciprocal"
    ) -> _NadaRational:
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
            on section 5.3 based on the Motzkin’s polynomial preprocessing technique. It uses the identity:

            .. math::
                \sigma(x) = \frac{1}{2}tanh(\frac{x}{2}) + \frac{1}{2}

            Note: stable for all input range as the approximation is truncated 
                    to 0/1 outside [-1, 1].

        "reciprocal" - computes sigmoid using :math:`1 + e^{-x}` and computing
            the reciprocal

            Note: stable for x in [-500, 500]. Unstable otherwise.

    Args:
        input (Union[Rational, SecretRational]): the input value.
        chebyshev_terms (int, optional): highest degree of Chebyshev polynomials.
                        Must be even and at least 6. Defaults to 32.
        method (str, optional): method used to compute sigmoid function. Defaults to "reciprocal".

    Returns:
        Union[Rational, SecretRational]: The sigmoid evaluation.

    Raises:
        ValueError: Raised if method type is not supported.
    """
    if method == "chebyshev":
        tanh_approx = tanh(input >> UnsignedInteger(1), method=method, chebyshev_terms=chebyshev_terms)
        return (tanh_approx >> UnsignedInteger(1)) + rational(0.5)
    elif method == "motzkin":
        tanh_approx = tanh(input >> UnsignedInteger(1), method=method, chebyshev_terms=chebyshev_terms)
        return (tanh_approx >> UnsignedInteger(1)) + rational(0.5)
    elif method == "reciprocal":
        # ltz is used for absolute value of input and to generate 'result'. 
        # We don't use 'abs' function to avoid computing ltz twice.
        ltz_cond = input < rational(0)
        ltz = ltz_cond.if_else(rational(1), rational(0))
        # compute absolute value of input
        sgn =  rational(1) - rational(2) * ltz
        pos_input = input * sgn 

        denominator = exp(-pos_input) + rational(1)
        pos_output = reciprocal(denominator, all_pos=True, initial=rational(0.75), iterations=3, exp_iters=9)

        # result is equivalent to (1 - ltz).if_else(pos_output, 1 - pos_output)
        result = pos_output + ltz - rational(2) * pos_output * ltz
        return result
    else:
        raise ValueError(f"Unrecognized method {method} for sigmoid")
    
def gelu(
        input: _NadaRational, 
        method: str = "tanh", 
        tanh_method: str = "reciprocal"
    ) -> _NadaRational:
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
        input (Union[Rational, SecretRational]): the input value.
        method (str, optional): method used to compute gelu function. Defaults to "tanh".
        tanh_method (str, optional): method used for tanh function. Defaults to "reciprocal".

    Returns:
        Union[Rational, SecretRational]: The gelu evaluation.

    Raises:
        ValueError: Raised if method type is not supported.
    """

    if method == "tanh":
        # Using common approximation:
        #       x/2 * (1 + tanh(0.797884560 * ( x + 0.04471 * x ** 3 ) ) )
        # where 0.797884560 ~= sqrt(2/pi).
        val = rational(0.797884560) * (input + rational(0.044715) * input ** 3)
        return (input * (rational(1) + tanh( val , method=tanh_method))) >> UnsignedInteger(1)
    elif method == "motzkin":
        # Using approximation from "BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers"
        # section 5.2 based on the Motzkin’s polynomial preprocessing technique.

        # ltz is used for absolute value of input and to compute relu. 
        # We don't use 'abs' and '_relu' functions to avoid computing ltz twice.
        ltz_cond = input < rational(0)
        ltz = ltz_cond.if_else(rational(1), rational(0))
        # absolute value
        sgn =  rational(1) - rational(2) * ltz
        abs_input = input * sgn
        # relu
        relu = input * (rational(1) - ltz)

        # Motzkin’s polynomial preprocessing
        g0 = rational(0.14439048359960427)
        g1 = rational(-0.7077117131613893)
        g2 = rational(4.5702822654246535)
        g3 = rational(-8.15444702051307)
        g4 = rational(16.382265425072532)
        gelu_p0 = (g0 * abs_input + g1) * abs_input + g2
        gelu_p1 = (gelu_p0 + g0 * abs_input + g3) * gelu_p0 + g4 + (input >> UnsignedInteger(1))

        cond_2_7 = abs_input > rational(2.7)
        result = cond_2_7.if_else(relu, gelu_p1)

        return result
    else:
        raise ValueError(f"Unrecognized method {method} for gelu")
    
def silu(
        input: _NadaRational, 
        method_sigmoid: str = "reciprocal", 
    ) -> _NadaRational:
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
            on section 5.3 based on the Motzkin’s polynomial preprocessing technique. It uses the identity:

            .. math::
                \sigma(x) = \frac{1}{2}tanh(\frac{x}{2}) + \frac{1}{2}

            Note: stable for all input range as the approximation is truncated 
                    to 0/1 outside [-1, 1].

        "reciprocal" - computes sigmoid using :math:`1 + e^{-x}` and computing
            the reciprocal

            Note: stable for x in [-500, 500]. Unstable otherwise.

    Args:
        input (Union[Rational, SecretRational]): the input value.
        method_sigmoid (str, optional): method used to compute sigmoid function. Defaults to "reciprocal".

    Returns:
        Union[Rational, SecretRational]: The sigmoid evaluation.

    Raises:
        ValueError: Raised if sigmoid method type is not supported.
    """
    silu = input * sigmoid(input, method=method_sigmoid)
    return silu
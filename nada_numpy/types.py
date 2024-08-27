"""Additional special data types"""

# pylint:disable=too-many-lines

import functools
import warnings
from typing import List, Optional, Tuple, Union

import nada_dsl as dsl
import numpy as np
from nada_dsl import (Input, Integer, Party, PublicInteger,
                      PublicUnsignedInteger, SecretInteger,
                      SecretUnsignedInteger, UnsignedInteger)

_NadaRational = Union["Rational", "SecretRational"]

_NadaType = Union[
    Integer,
    PublicInteger,
    PublicUnsignedInteger,
    SecretInteger,
    SecretUnsignedInteger,
    UnsignedInteger,
]


class SecretBoolean(dsl.SecretBoolean):
    """SecretBoolean rational wrapper"""

    def __init__(self, value: dsl.SecretBoolean) -> None:
        """
        Initialization.

        Args:
            value (dsl.SecretBoolean): SecretBoolean value.
        """
        super().__init__(value.inner)

    def if_else(
        self,
        true: Union[_NadaType, "SecretRational", "Rational"],
        false: Union[_NadaType, "SecretRational", "Rational"],
    ) -> Union[SecretInteger, SecretUnsignedInteger, "SecretRational"]:
        """
        If-else logic. If the boolean is True, true is returned. If not, false is returned.

        Args:
            true (Union[_NadaType, SecretRational, Rational]): First argument.
            false (Union[_NadaType, SecretRational, Rational]): Second argument.

        Raises:
            ValueError: Raised when incompatibly-scaled values are passed.
            TypeError: Raised when invalid operation is called.

        Returns:
            Union[SecretInteger, SecretUnsignedInteger, "SecretRational"]: Return value.
        """
        first_arg = true
        second_arg = false
        if isinstance(true, (SecretRational, Rational)) and isinstance(
            false, (SecretRational, Rational)
        ):
            # Both are SecretRational or Rational objects
            if true.log_scale != false.log_scale:
                raise ValueError("Cannot output values with different scales.")
            first_arg = true.value
            second_arg = false.value
        elif isinstance(true, (Rational, SecretRational)) or isinstance(
            false, (Rational, SecretRational)
        ):
            # Both are SecretRational or Rational objects
            raise TypeError(f"Invalid operation: {self}.IfElse({true}, {false})")

        result = super().if_else(first_arg, second_arg)

        if isinstance(true, (SecretRational, Rational)):
            # If we have a SecretBoolean, the return type will be SecretInteger,
            # thus promoted to SecretRational
            return SecretRational(result, true.log_scale, is_scaled=True)
        return result


class PublicBoolean(dsl.PublicBoolean):
    """PublicBoolean rational wrapper"""

    def __init__(self, value: dsl.PublicBoolean) -> None:
        """
        Initialization.

        Args:
            value (dsl.PublicBoolean): PublicBoolean value.
        """
        super().__init__(value.inner)

    def if_else(
        self,
        true: Union[_NadaType, "SecretRational", "Rational"],
        false: Union[_NadaType, "SecretRational", "Rational"],
    ) -> Union[
        PublicInteger,
        PublicUnsignedInteger,
        SecretInteger,
        SecretUnsignedInteger,
        "Rational",
        "SecretRational",
    ]:
        """
        If-else logic. If the boolean is True, true is returned. If not, false is returned.

        Args:
            true (Union[_NadaType, SecretRational, Rational]): First argument.
            false (Union[_NadaType, SecretRational, Rational]): Second argument.

        Raises:
            ValueError: Raised when incompatibly-scaled values are passed.
            TypeError: Raised when invalid operation is called.

        Returns:
            Union[PublicInteger, PublicUnsignedInteger, SecretInteger,
                SecretUnsignedInteger, "Rational", "SecretRational"]: Return value.
        """
        first_arg = true
        second_arg = false
        if isinstance(true, (SecretRational, Rational)) and isinstance(
            false, (SecretRational, Rational)
        ):
            # Both are SecretRational or Rational objects
            if true.log_scale != false.log_scale:
                raise ValueError("Cannot output values with different scales.")
            first_arg = true.value
            second_arg = false.value
        elif isinstance(true, (Rational, SecretRational)) or isinstance(
            false, (Rational, SecretRational)
        ):
            # Both are SecretRational or Rational objects but of different type
            raise TypeError(f"Invalid operation: {self}.IfElse({true}, {false})")

        result = super().if_else(first_arg, second_arg)

        if isinstance(true, SecretRational) or isinstance(false, SecretRational):
            return SecretRational(result, true.log_scale, is_scaled=True)
        if isinstance(true, Rational) and isinstance(false, Rational):
            return Rational(result, true.log_scale, is_scaled=True)
        return result


class Rational:  # pylint:disable=too-many-public-methods
    """Wrapper class to store scaled Integer values representing a fixed-point number."""

    def __init__(
        self,
        value: Union[Integer, PublicInteger],
        log_scale: Optional[int] = None,
        is_scaled: bool = True,
    ) -> None:
        """
        Initializes wrapper around Integer object.

        Args:
            value (Union[Integer, PublicInteger]): The value to be representedas a Rational.
            log_scale (int, optional): Quantization scaling factor.
                Defaults to RationalConfig.log_scale.
            is_scaled (bool, optional): Flag that represents whether the value is already scaled.
                Defaults to True.

        Raises:
            TypeError: If value is of an incompatible type.
        """
        if not isinstance(value, (Integer, PublicInteger)):
            raise TypeError(f"Cannot instantiate Rational from type `{type(value)}`.")

        if log_scale is None:
            log_scale = get_log_scale()
        self._log_scale = log_scale

        if is_scaled is False:
            value = value * Integer(
                1 << log_scale
            )  # TODO: replace with shift when supported
        self._value = value

    @property
    def log_scale(self) -> int:
        """
        Getter for the logarithmic scale value.

        Returns:
            int: Logarithmic scale value.
        """
        return self._log_scale

    @property
    def value(self) -> Union[Integer, PublicInteger]:
        """
        Getter for the underlying Integer value.

        Returns:
            Union[Integer, PublicInteger]: The Integer value.
        """
        return self._value

    def add(
        self, other: _NadaRational, ignore_scale: bool = False
    ) -> Union["Rational", "SecretRational"]:
        """
        Add two rational numbers.

        Args:
            other (_NadaRational): Other rational number to add.
            ignore_scale (bool, optional): Flag to disable scale checking. Disabling
                auto-scaling can lead to significant performance gains as it allows
                "bundling" scaling ops. However, it is advanced feature and can lead
                to unexpected results if used incorrectly. Defaults to False.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.

        Raises:
            TypeError: If the other value is of an incompatible type.
            ValueError: If the log scales of the two values are different.
        """
        if not isinstance(other, (Rational, SecretRational)):
            return NotImplemented

        if not ignore_scale and self.log_scale != other.log_scale:
            raise ValueError("Cannot add values with different scales.")

        if isinstance(other, SecretRational):
            return SecretRational(
                other.value + self.value, self.log_scale, is_scaled=True
            )
        return Rational(self.value + other.value, self.log_scale, is_scaled=True)

    def __add__(self, other: _NadaRational) -> Union["Rational", "SecretRational"]:
        """
        Add two rational numbers.

        Args:
            other (_NadaRational): Other rational number to add.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.
        """
        return self.add(other)

    def __iadd__(self, other: _NadaRational) -> Union["Rational", "SecretRational"]:
        """
        Add two rational numbers inplace.

        Args:
            other (_NadaRational): Other rational number to add.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.
        """
        return self.add(other)

    def sub(
        self, other: _NadaRational, ignore_scale: bool = False
    ) -> Union["Rational", "SecretRational"]:
        """
        Subtract two rational numbers.

        Args:
            other (_NadaRational): Other rational number to subtract.

        Returns:
            Union[Rational, SecretRational]: Result of the subtraction.
            ignore_scale (bool, optional): Flag to disable scale checking. Disabling
                auto-scaling can lead to significant performance gains as it allows
                "bundling" scaling ops. However, it is advanced feature and can lead
                to unexpected results if used incorrectly. Defaults to False.

        Raises:
            TypeError: If the other value is of an incompatible type.
            ValueError: If the log scales of the two values are different.
        """
        if not isinstance(other, (Rational, SecretRational)):
            return NotImplemented

        if not ignore_scale and self.log_scale != other.log_scale:
            raise ValueError("Cannot substract values with different scales.")

        if isinstance(other, SecretRational):
            return SecretRational(
                self.value - other.value, self.log_scale, is_scaled=True
            )
        return Rational(self.value - other.value, self.log_scale, is_scaled=True)

    def __sub__(self, other: _NadaRational) -> Union["Rational", "SecretRational"]:
        """
        Subtract two rational numbers.

        Args:
            other (_NadaRational): Other rational number to add.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.
        """
        return self.sub(other)

    def __isub__(self, other: _NadaRational) -> Union["Rational", "SecretRational"]:
        """
        Subtract two rational numbers inplace.

        Args:
            other (_NadaRational): Other rational number to add.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.
        """
        return self.sub(other)

    def mul_no_rescale(
        self, other: _NadaRational, ignore_scale: bool = False
    ) -> Union["Rational", "SecretRational"]:
        """
        Multiply two rational numbers.

        WARNING: This function does not rescale by default. Use `mul` to multiply and rescale.

        Args:
            other (_NadaRational): Other rational number to multiply.
            ignore_scale (bool, optional): Flag to disable scale checking. Disabling
                auto-scaling can lead to significant performance gains as it allows
                "bundling" scaling ops. However, it is advanced feature and can lead
                to unexpected results if used incorrectly. Defaults to False.

        Returns:
            Union[Rational, SecretRational]: Result of the multiplication.

        Raises:
            TypeError: If the other value is of an incompatible type.
            ValueError: If the log scales of the two values are different.
        """
        if not isinstance(other, (Rational, SecretRational)):
            return NotImplemented

        if not ignore_scale and self.log_scale != other.log_scale:
            raise ValueError("Cannot multiply values with different scales.")

        if isinstance(other, SecretRational):
            return SecretRational(
                self.value * other.value,
                self.log_scale + other.log_scale,
                is_scaled=True,
            )
        return Rational(
            self.value * other.value,
            self.log_scale + other.log_scale,
            is_scaled=True,
        )

    def mul(
        self, other: _NadaRational, ignore_scale: bool = False
    ) -> Union["Rational", "SecretRational"]:
        """
        Multiply two rational numbers and rescale the result.

        Args:
            other (_NadaRational): Other rational number to multiply.
            ignore_scale (bool, optional): Flag to disable scale checking. Disabling
                auto-scaling can lead to significant performance gains as it allows
                "bundling" scaling ops. However, it is advanced feature and can lead
                to unexpected results if used incorrectly. Defaults to False.

        Returns:
            Union[Rational, SecretRational]: Result of the multiplication.
        """
        c = self.mul_no_rescale(other, ignore_scale=ignore_scale)
        d = c.rescale_down()
        return d

    def __mul__(self, other: _NadaRational) -> Union["Rational", "SecretRational"]:
        """
        Multiply two rational numbers.

        Args:
            other (_NadaRational): Other rational number to add.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.
        """
        return self.mul(other)

    def __imul__(self, other: _NadaRational) -> Union["Rational", "SecretRational"]:
        """
        Multiply two rational numbers inplace.

        Args:
            other (_NadaRational): Other rational number to add.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.
        """
        return self.mul(other)

    def divide_no_rescale(
        self, other: _NadaRational, ignore_scale: bool = False
    ) -> Union["Rational", "SecretRational"]:
        """
        Divide two rational numbers.

        Args:
            other (_NadaRational): Other rational number to divide by.

        Returns:
            Union[Rational, SecretRational]: Result of the division.

        Raises:
            TypeError: If the other value is of an incompatible type.
            ValueError: If the log scales of the two values are different.
        """
        if not isinstance(other, (Rational, SecretRational)):
            return NotImplemented

        if not ignore_scale and self.log_scale != other.log_scale + get_log_scale():
            raise ValueError(
                f"Cannot divide values where scale is: {self.log_scale} / {other.log_scale}."
                f"Required scale: {self.log_scale}  / {other.log_scale + get_log_scale()}"
            )

        if isinstance(other, SecretRational):
            return SecretRational(
                self.value / other.value,
                self.log_scale - other.log_scale,
                is_scaled=True,
            )
        return Rational(
            self.value / other.value,
            self.log_scale - other.log_scale,
            is_scaled=True,
        )

    def divide(
        self, other: _NadaRational, ignore_scale: bool = False
    ) -> Union["Rational", "SecretRational"]:
        """
        Divide two rational numbers and rescale the result.

        Args:
            other (_NadaRational): Other rational number to divide by.
            ignore_scale (bool, optional): Flag to disable scale checking. Disabling
                auto-scaling can lead to significant performance gains as it allows
                "bundling" scaling ops. However, it is advanced feature and can lead
                to unexpected results if used incorrectly. Defaults to False.

        Returns:
            Union[Rational, SecretRational]: Result of the division.
        """
        a = self.rescale_up()
        c = a.divide_no_rescale(other, ignore_scale)
        return c

    def __truediv__(self, other: _NadaRational) -> Union["Rational", "SecretRational"]:
        """
        Divide two rational numbers.

        Args:
            other (_NadaRational): Other rational number to add.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.
        """
        return self.divide(other)

    def __itruediv__(self, other: _NadaRational) -> Union["Rational", "SecretRational"]:
        """
        Divide two rational numbers inplace.

        Args:
            other (_NadaRational): Other rational number to add.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.
        """
        return self.divide(other)

    def __pow__(self, other: int) -> "Rational":
        """
        Raise a rational number to an integer power using binary exponentiation.

        Args:
            other (int): The exponent.

        Returns:
            Rational: Result of the power operation.

        Raises:
            TypeError: If the exponent is not an integer.
        """
        if not isinstance(other, int):
            raise TypeError(f"Cannot raise Rational to a power of type `{type(other)}`")

        result = Rational(Integer(1), self.log_scale, is_scaled=False)

        if other == 0:
            return result  # Any number to the power of 0 is 1

        base = self

        exponent = abs(other)
        while exponent > 0:
            if exponent % 2 == 1:
                result = result * base  # type: ignore
            base *= base  # type: ignore
            exponent //= 2

        if other < 0:
            return rational(1) / Rational(  # type: ignore
                result.value, result.log_scale, is_scaled=True
            )

        return result

    def __neg__(self) -> "Rational":
        """
        Negate the Rational value.

        Returns:
            Rational: Negated Rational value.
        """
        return Rational(self.value * Integer(-1), self.log_scale, is_scaled=True)

    def __lshift__(self, other: UnsignedInteger) -> "Rational":
        """
        Left shift the Rational value.

        Args:
            other (UnsignedInteger): The value to left shift by.

        Returns:
            Rational: Left shifted Rational value.
        """
        return Rational(self.value << other, self.log_scale)

    def __rshift__(self, other: UnsignedInteger) -> "Rational":
        """
        Right shift the Rational value.

        Args:
            other (UnsignedInteger): The value to right shift by.

        Returns:
            Rational: Right shifted Rational value.
        """
        return Rational(self.value >> other, self.log_scale)

    def __lt__(self, other: _NadaRational) -> Union[PublicBoolean, SecretBoolean]:
        """
        Check if this Rational is less than another.

        Args:
            other (_NadaRational): The other value to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            Union[PublicBoolean, SecretBoolean]: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        if isinstance(other, SecretRational):
            return SecretBoolean(self.value < other.value)
        return PublicBoolean(self.value < other.value)

    def __gt__(self, other: _NadaRational) -> Union[PublicBoolean, SecretBoolean]:
        """
        Check if this Rational is greater than another.

        Args:
            other (_NadaRational): The other value to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            Union[PublicBoolean, SecretBoolean]: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        if isinstance(other, SecretRational):
            return SecretBoolean(self.value > other.value)
        return PublicBoolean(self.value > other.value)

    def __le__(self, other: _NadaRational) -> Union[PublicBoolean, SecretBoolean]:
        """
        Check if this Rational is less than or equal to another.

        Args:
            other (_NadaRational): The other value to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            Union[PublicBoolean, SecretBoolean]: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        if isinstance(other, SecretRational):
            return SecretBoolean(self.value <= other.value)
        return PublicBoolean(self.value <= other.value)

    def __ge__(self, other: _NadaRational) -> Union[PublicBoolean, SecretBoolean]:
        """
        Check if this Rational is greater than or equal to another.

        Args:
            other (_NadaRational): The other value to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            Union[PublicBoolean, SecretBoolean]: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        if isinstance(other, SecretRational):
            return SecretBoolean(self.value >= other.value)
        return PublicBoolean(self.value >= other.value)

    def __eq__(self, other: _NadaRational) -> Union[PublicBoolean, SecretBoolean]:  # type: ignore
        """
        Check if this Rational is equal to another.

        Args:
            other (_NadaRational): The other value to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            Union[PublicBoolean, SecretBoolean]: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        if isinstance(other, SecretRational):
            return SecretBoolean(self.value == other.value)
        return PublicBoolean(self.value == other.value)

    def __ne__(self, other: _NadaRational) -> Union[PublicBoolean, SecretBoolean]:  # type: ignore
        """
        Check if this Rational is not equal to another.

        Args:
            other (_NadaRational): The other value to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            Union[PublicBoolean, SecretBoolean]: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        if isinstance(other, SecretRational):
            return SecretBoolean(self.value != other.value)
        return PublicBoolean(self.value != other.value)

    def rescale_up(self, log_scale: Optional[int] = None) -> "Rational":
        """
        Rescale the value in the upward direction by a scaling factor.

        This is equivalent to multiplying the value by `2**(log_scale)`.

        Args:
            log_scale (int, optional): Scaling factor to rescale the value.
                Defaults to RationalConfig.log_scale.

        Returns:
            Rational: Rescaled Rational value.
        """
        if log_scale is None:
            log_scale = get_log_scale()

        return Rational(
            self._value
            * Integer(1 << log_scale),  # TODO: replace with shift when supported
            self.log_scale + log_scale,
            is_scaled=True,
        )

    def rescale_down(self, log_scale: Optional[int] = None) -> "Rational":
        """
        Rescale the value in the downward direction by a scaling factor.

        This is equivalent to dividing the value by `2**(log_scale)`.

        Args:
            log_scale (int, optional): Scaling factor to rescale the value.
                Defaults to RationalConfig.log_scale.

        Returns:
            Rational: Rescaled Rational value.
        """
        if log_scale is None:
            log_scale = get_log_scale()

        return Rational(
            self._value
            / Integer(1 << log_scale),  # TODO: replace with shift when supported
            self.log_scale - log_scale,
            is_scaled=True,
        )

    # Non-linear functions

    def sign(self) -> "Rational":
        """Computes the sign value (0 is considered positive)"""

        result = sign(self)
        if not isinstance(result, Rational):
            raise TypeError("sign input should be of type Rational.")
        return result

    def abs(self) -> "Rational":
        """Computes the absolute value"""

        result = fxp_abs(self)
        if not isinstance(result, Rational):
            raise TypeError("abs input should be of type Rational.")
        return result

    def exp(self, iterations: int = 8) -> "Rational":
        """
        Approximates the exponential function using a limit approximation.

        The exponential function is approximated using the following limit:

            exp(x) = lim_{n -> ∞} (1 + x / n) ^ n

        The function is computed by choosing n = 2 ** d, where d is set to `iterations`.
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
            Rational: The approximated value of the exponential function.
        """

        result = exp(self, iterations=iterations)
        if not isinstance(result, Rational):
            raise TypeError("exp input should be of type Rational.")
        return result

    def polynomial(self, coefficients: list) -> "Rational":
        """
        Computes a polynomial function on a value with given coefficients.

        The coefficients can be provided as a list of values.
        They should be ordered from the linear term (order 1) first, ending with the
        highest order term.
        **Note: The constant term is not included.**

        Args:
            coefficients (list): The coefficients of the polynomial, ordered by increasing degree.

        Returns:
            Rational: The result of the polynomial function applied to the input x.
        """

        result = polynomial(self, coefficients=coefficients)
        if not isinstance(result, Rational):
            raise TypeError("polynomial input should be of type Rational.")
        return result

    def log(
        self,
        input_in_01: bool = False,
        iterations: int = 2,
        exp_iterations: int = 8,
        order: int = 8,
    ) -> "Rational":
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
            exp_iterations (int, optional): Number of iterations for the limit approximation
                of exp. Defaults to 8.
            order (int, optional): Number of polynomial terms used (order of Householder
                approximation). Defaults to 8.

        Returns:
            Rational: The approximate value of the natural logarithm.
        """

        result = log(
            self,
            input_in_01=input_in_01,
            iterations=iterations,
            exp_iterations=exp_iterations,
            order=order,
        )
        if not isinstance(result, Rational):
            raise TypeError("log input should be of type Rational.")
        return result

    def reciprocal(  # pylint: disable=too-many-arguments
        self,
        all_pos: bool = False,
        initial: Optional["Rational"] = None,
        input_in_01: bool = False,
        iterations: int = 10,
        log_iters: int = 1,
        exp_iters: int = 8,
        method: str = "NR",
    ) -> "Rational":
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
            Rational: The approximate value of the reciprocal

        .. _Newton-Raphson:
            https://en.wikipedia.org/wiki/Newton%27s_method
        """

        result = reciprocal(
            self,
            all_pos=all_pos,
            initial=initial,
            input_in_01=input_in_01,
            iterations=iterations,
            log_iters=log_iters,
            exp_iters=exp_iters,
            method=method,
        )
        if not isinstance(result, Rational):
            raise TypeError("reciprocal input should be of type Rational.")
        return result

    def inv_sqrt(
        self,
        initial: Optional["Rational"] = None,
        iterations: int = 5,
        method: str = "NR",
    ) -> "Rational":
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
            initial (Union[Rational, None], optional): sets the initial value for the
                        Newton-Raphson iterations. By default, this will be set to allow the
                        method to converge over a fairly large domain.
            iterations (int, optional): determines the number of Newton-Raphson iterations to run.
            method (str, optional): method used to compute inv_sqrt. Defaults to "NR".

        Returns:
            Rational: The approximate value of the inv_sqrt.

        .. _Newton-Raphson:
            https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
        """

        result = inv_sqrt(self, initial=initial, iterations=iterations, method=method)
        if not isinstance(result, Rational):
            raise TypeError("inv_sqrt input should be of type Rational.")
        return result

    def sqrt(
        self,
        initial: Optional["Rational"] = None,
        iterations: int = 5,
        method: str = "NR",
    ) -> "Rational":
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
            initial (Union[Rational, None], optional): sets the initial value for the inverse
                square root Newton-Raphson iterations. By default, this will be set to allow
                convergence over a fairly large domain. Defaults to None.
            iterations (int, optional):  determines the number of Newton-Raphson iterations to run.
                Defaults to 5.
            method (str, optional): method used to compute sqrt. Defaults to "NR".

        Returns:
            Rational: The approximate value of the sqrt.

        .. _Newton-Raphson:
            https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
        """

        result = sqrt(self, initial=initial, iterations=iterations, method=method)
        if not isinstance(result, Rational):
            raise TypeError("sqrt input should be of type Rational.")
        return result

    # Trigonometry

    def cossin(self, iterations: int = 10) -> Tuple["Rational", "Rational"]:
        r"""Computes cosine and sine through e^(i * input) where i is the imaginary unit through
        the formula:

        .. math::
            Re\{e^{i * input}\}, Im\{e^{i * input}\} = \cos(input), \sin(input)

        Args:
            iterations (int, optional): determines the number of iterations to run. Defaults to 10.

        Returns:
            Tuple[Rational, Rational]:
                A tuple where the first element is cos and the second element is the sin.
        """

        result = cossin(self, iterations=iterations)
        if not isinstance(result, Rational):
            raise TypeError("cossin input should be of type Rational.")
        return result

    def cos(self, iterations: int = 10) -> "Rational":
        r"""Computes the cosine of the input using cos(x) = Re{exp(i * x)}.

        Note: unstable outside [-30, 30]

        Args:
            iterations (int, optional): determines the number of iterations to run. Defaults to 10.

        Returns:
            Rational: The approximate value of the cosine.
        """

        result = cos(self, iterations=iterations)
        if not isinstance(result, Rational):
            raise TypeError("cos input should be of type Rational.")
        return result

    def sin(self, iterations: int = 10) -> "Rational":
        r"""Computes the sine of the input using sin(x) = Im{exp(i * x)}.

        Note: unstable outside [-30, 30]

        Args:
            iterations (int, optional): determines the number of iterations to run. Defaults to 10.

        Returns:
            Rational: The approximate value of the sine.
        """

        result = sin(self, iterations=iterations)
        if not isinstance(result, Rational):
            raise TypeError("sin input should be of type Rational.")
        return result

    def tan(self, iterations: int = 10) -> "Rational":
        r"""Computes the tan of the input using tan(x) = sin(x) / cos(x).

        Note: unstable outside [-30, 30]

        Args:
            iterations (int, optional): determines the number of iterations to run. Defaults to 10.

        Returns:
            Rational: The approximate value of the tan.
        """

        result = tan(self, iterations=iterations)
        if not isinstance(result, Rational):
            raise TypeError("tan input should be of type Rational.")
        return result

    # Activation functions

    def tanh(self, chebyshev_terms: int = 32, method: str = "reciprocal") -> "Rational":
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
            Rational: The tanh evaluation.

        Raises:
            ValueError: Raised if method type is not supported.
        """

        result = tanh(self, chebyshev_terms=chebyshev_terms, method=method)
        if not isinstance(result, Rational):
            raise TypeError("tanh input should be of type Rational.")
        return result

    def sigmoid(
        self, chebyshev_terms: int = 32, method: str = "reciprocal"
    ) -> "Rational":
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
            method (str, optional): method used to compute sigmoid function.
                Defaults to "reciprocal".

        Returns:
            Rational: The sigmoid evaluation.

        Raises:
            ValueError: Raised if method type is not supported.
        """

        result = sigmoid(self, chebyshev_terms=chebyshev_terms, method=method)
        if not isinstance(result, Rational):
            raise TypeError("sigmoid input should be of type Rational.")
        return result

    def gelu(self, method: str = "tanh", tanh_method: str = "reciprocal") -> "Rational":
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
            Rational: The gelu evaluation.

        Raises:
            ValueError: Raised if method type is not supported.
        """

        result = gelu(self, method=method, tanh_method=tanh_method)
        if not isinstance(result, Rational):
            raise TypeError("gelu input should be of type Rational.")
        return result

    def silu(
        self,
        method_sigmoid: str = "reciprocal",
    ) -> "Rational":
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
                on section 5.3 based on the Motzkin’s polynomial preprocessing technique.
                It uses the identity:

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
            Rational: The sigmoid evaluation.

        Raises:
            ValueError: Raised if sigmoid method type is not supported.
        """

        if method_sigmoid is None:
            method_sigmoid = "reciprocal"

        result = silu(self, method_sigmoid=method_sigmoid)
        if not isinstance(result, Rational):
            raise TypeError("silu input should be of type Rational.")
        return result


class SecretRational:  # pylint:disable=too-many-public-methods
    """Wrapper class to store scaled SecretInteger values representing a fixed-point number."""

    def __init__(
        self,
        value: SecretInteger,
        log_scale: Optional[int] = None,
        is_scaled: bool = True,
    ) -> None:
        """
        Initializes wrapper around SecretInteger object.
        The object should come scaled up by default otherwise precision may be lost.

        Args:
            value (SecretInteger): SecretInteger input value.
            log_scale (int, optional): Quantization scaling factor.
                Defaults to RationalConfig.log_scale.
            is_scaled (bool, optional): Flag that indicates whether provided value has already been
                scaled by log_scale factor. Defaults to True.

        Raises:
            TypeError: If value is of an incompatible type.
        """
        if not isinstance(value, SecretInteger):
            raise TypeError(
                f"Cannot instantiate SecretRational from type `{type(value)}`."
            )

        if log_scale is None:
            log_scale = get_log_scale()
        self._log_scale = log_scale

        if is_scaled is False:
            value = value << UnsignedInteger(log_scale)
        self._value = value

    @property
    def log_scale(self) -> int:
        """
        Getter for the logarithmic scale value.

        Returns:
            int: Logarithmic scale value.
        """
        return self._log_scale

    @property
    def value(self) -> SecretInteger:
        """
        Getter for the underlying SecretInteger value.

        Returns:
            SecretInteger: The SecretInteger value.
        """
        return self._value

    def add(self, other: _NadaRational, ignore_scale: bool = False) -> "SecretRational":
        """
        Add two SecretRational numbers.

        Args:
            other (_NadaRational): The other SecretRational to add.
            ignore_scale (bool, optional): Flag to disable scale checking. Disabling
                auto-scaling can lead to significant performance gains as it allows
                "bundling" scaling ops. However, it is advanced feature and can lead
                to unexpected results if used incorrectly. Defaults to False.

        Raises:
            TypeError: If the other value is of an incompatible type.
            ValueError: If the log scales of the two values are different.

        Returns:
            SecretRational: Result of the addition.
        """
        if not isinstance(other, (Rational, SecretRational)):
            return NotImplemented

        if not ignore_scale and self.log_scale != other.log_scale:
            raise ValueError("Cannot add values with different scales.")

        return SecretRational(self.value + other.value, self.log_scale)

    def __add__(self, other: _NadaRational) -> "SecretRational":
        """
        Add two secret rational numbers.

        Args:
            other (_NadaRational): Other rational number to add.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.
        """
        return self.add(other)

    def __iadd__(self, other: _NadaRational) -> "SecretRational":
        """
        Add two secret rational numbers inplace.

        Args:
            other (_NadaRational): Other rational number to add.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.
        """
        return self.add(other)

    def sub(self, other: _NadaRational, ignore_scale: bool = False) -> "SecretRational":
        """
        Subtract two SecretRational numbers.

        Args:
            other (_NadaRational): The other SecretRational to subtract.
            ignore_scale (bool, optional): Flag to disable scale checking. Disabling
                auto-scaling can lead to significant performance gains as it allows
                "bundling" scaling ops. However, it is advanced feature and can lead
                to unexpected results if used incorrectly. Defaults to False.

        Raises:
            TypeError: If the other value is of an incompatible type.
            ValueError: If the log scales of the two values are different.

        Returns:
            SecretRational: Result of the subtraction.
        """
        if not isinstance(other, (Rational, SecretRational)):
            return NotImplemented

        if not ignore_scale and self.log_scale != other.log_scale:
            raise ValueError("Cannot substract values with different scales.")

        return SecretRational(self.value - other.value, self.log_scale)

    def __sub__(self, other: _NadaRational) -> "SecretRational":
        """
        Subtract two secret rational numbers.

        Args:
            other (_NadaRational): Other rational number to add.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.
        """
        return self.sub(other)

    def __isub__(self, other: _NadaRational) -> "SecretRational":
        """
        Subtract two secret rational numbers inplace.

        Args:
            other (_NadaRational): Other rational number to add.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.
        """
        return self.sub(other)

    def mul_no_rescale(
        self, other: _NadaRational, ignore_scale: bool = False
    ) -> "SecretRational":
        """
        Multiply two SecretRational numbers.

        Args:
            other (_NadaRational): The other SecretRational to multiply.
            ignore_scale (bool, optional): Flag to disable scale checking. Disabling
                auto-scaling can lead to significant performance gains as it allows
                "bundling" scaling ops. However, it is advanced feature and can lead
                to unexpected results if used incorrectly. Defaults to False.

        Raises:
            TypeError: If the other value is of an incompatible type.
            ValueError: If the log scales of the two values are different.

        Returns:
            SecretRational: Result of the multiplication.
        """
        if not isinstance(other, (Rational, SecretRational)):
            return NotImplemented

        if not ignore_scale and self.log_scale != other.log_scale:
            raise ValueError("Cannot multiply values with different scales.")

        return SecretRational(
            self.value * other.value, self.log_scale + other.log_scale
        )

    def mul(self, other: _NadaRational, ignore_scale: bool = False) -> "SecretRational":
        """
        Multiply two SecretRational numbers and rescale the result.

        Args:
            other (_NadaRational): The other SecretRational to multiply.
            ignore_scale (bool, optional): Flag to disable scale checking. Disabling
                auto-scaling can lead to significant performance gains as it allows
                "bundling" scaling ops. However, it is advanced feature and can lead
                to unexpected results if used incorrectly. Defaults to False.

        Returns:
            SecretRational: Result of the multiplication, rescaled.
        """
        c = self.mul_no_rescale(other, ignore_scale=ignore_scale)
        if c is NotImplemented:
            # Note that, because this function would be executed under a NadaArray,
            # the NotImplemented value will be handled by the caller (in principle NadaArray)
            # The caller will then call the mul function of the NadaArray
            # The broadcasting will execute element-wise multiplication,
            # so rescale_down will be taken care by that function
            return c
        d = c.rescale_down()
        return d

    def __mul__(self, other: _NadaRational) -> "SecretRational":
        """
        Multiply two secret rational numbers.

        Args:
            other (_NadaRational): Other rational number to add.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.
        """
        return self.mul(other)

    def __imul__(self, other: _NadaRational) -> "SecretRational":
        """
        Multiply two secret rational numbers inplace.

        Args:
            other (_NadaRational): Other rational number to add.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.
        """
        return self.mul(other)

    def divide_no_rescale(
        self, other: _NadaRational, ignore_scale: bool = False
    ) -> "SecretRational":
        """
        Divide two SecretRational numbers.

        Args:
            other (_NadaRational): The other SecretRational to divide by.
            ignore_scale (bool, optional): Flag to disable scale checking. Disabling
                auto-scaling can lead to significant performance gains as it allows
                "bundling" scaling ops. However, it is advanced feature and can lead
                to unexpected results if used incorrectly. Defaults to False.

        Raises:
            TypeError: If the other value is of an incompatible type.
            ValueError: If the log scales of the two values are different.

        Returns:
            SecretRational: Result of the division.
        """
        if not isinstance(other, (Rational, SecretRational)):
            return NotImplemented

        if not ignore_scale and self.log_scale != other.log_scale + get_log_scale():
            raise ValueError(
                f"Cannot divide values where scale is: {self.log_scale} / {other.log_scale}."
                f"Required scale: {self.log_scale}  / {other.log_scale + get_log_scale()}"
            )

        return SecretRational(
            self.value / other.value, self.log_scale - other.log_scale
        )

    def divide(
        self, other: _NadaRational, ignore_scale: bool = False
    ) -> "SecretRational":
        """
        Divide two SecretRational numbers and rescale the result.

        Args:
            other (_NadaRational): The other SecretRational to divide by.
            ignore_scale (bool, optional): Flag to disable scale checking. Disabling
                auto-scaling can lead to significant performance gains as it allows
                "bundling" scaling ops. However, it is advanced feature and can lead
                to unexpected results if used incorrectly. Defaults to False.

        Returns:
            SecretRational: Result of the division, rescaled.
        """
        # Note: If the other value is a NadaArray, the divide-no-rescale function will
        # return NotImplemented
        # This will cause that the divide function will return NotImplemented as well
        # The NotImplemented value will be handled by the caller (in principle NadaArray)
        # The caller will then call the divide function of the NadaArray
        # The rescale up, because there is no follow up, will not be taken into consideration.
        a = self.rescale_up()
        c = a.divide_no_rescale(other, ignore_scale=ignore_scale)
        return c

    def __truediv__(self, other: _NadaRational) -> "SecretRational":
        """
        Divide two secret rational numbers.

        Args:
            other (_NadaRational): Other rational number to add.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.
        """
        return self.divide(other)

    def __itruediv__(self, other: _NadaRational) -> "SecretRational":
        """
        Divide two secret rational numbers inplace.

        Args:
            other (_NadaRational): Other rational number to add.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.
        """
        return self.divide(other)

    def __pow__(self, other: int) -> Union["Rational", "SecretRational"]:
        """
        Raise a SecretRational to an integer power using binary exponentiation.

        Args:
            other (int): The exponent.

        Raises:
            TypeError: If the exponent is not an integer.

        Returns:
            Union[Rational, SecretRational]: Result of the power operation.
        """
        if not isinstance(other, int):
            raise TypeError(
                f"Cannot raise SecretRational to a power of type `{type(other)}`"
            )

        result = Rational(Integer(1), self.log_scale, is_scaled=False)

        if other == 0:
            return result  # Any number to the power of 0 is 1

        base = self

        exponent = abs(other)
        while exponent > 0:
            if exponent % 2 == 1:
                result = result * base  # type: ignore
            base *= base  # type: ignore
            exponent //= 2

        if other < 0:
            return rational(1) / SecretRational(  # type: ignore
                result.value, result.log_scale, is_scaled=True
            )

        return result

    def __neg__(self) -> "SecretRational":
        """
        Negate the SecretRational value.

        Returns:
            SecretRational: Negated SecretRational value.
        """
        return SecretRational(self.value * Integer(-1), self.log_scale)

    def __lshift__(self, other: UnsignedInteger) -> "SecretRational":
        """
        Left shift the SecretRational value.

        Args:
            other (UnsignedInteger): The value to left shift by.

        Returns:
            SecretRational: Left shifted SecretRational value.
        """
        return SecretRational(self.value << other, self.log_scale)

    def __rshift__(self, other: UnsignedInteger) -> "SecretRational":
        """
        Right shift the SecretRational value.

        Args:
            other (UnsignedInteger): The value to right shift by.

        Returns:
            SecretRational: Right shifted SecretRational value.
        """
        return SecretRational(self.value >> other, self.log_scale)

    def __lt__(self, other: _NadaRational) -> SecretBoolean:
        """
        Check if this SecretRational is less than another.

        Args:
            other (_NadaRational): The other SecretRational to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            SecretBoolean: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        return SecretBoolean(self.value < other.value)

    def __gt__(self, other: _NadaRational) -> SecretBoolean:
        """
        Check if this SecretRational is greater than another.

        Args:
            other (_NadaRational): The other SecretRational to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            SecretBoolean: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        return SecretBoolean(self.value > other.value)

    def __le__(self, other: _NadaRational) -> SecretBoolean:
        """
        Check if this SecretRational is less than or equal to another.

        Args:
            other (_NadaRational): The other SecretRational to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            SecretBoolean: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        return SecretBoolean(self.value <= other.value)

    def __ge__(self, other: _NadaRational) -> SecretBoolean:
        """
        Check if this SecretRational is greater than or equal to another.

        Args:
            other (_NadaRational): The other SecretRational to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            SecretBoolean: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        return SecretBoolean(self.value >= other.value)

    def __eq__(self, other: _NadaRational) -> SecretBoolean:  # type: ignore
        """
        Check if this SecretRational is equal to another.

        Args:
            other (_NadaRational): The other SecretRational to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            SecretBoolean: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        return SecretBoolean(self.value == other.value)

    def __ne__(self, other: _NadaRational) -> SecretBoolean:  # type: ignore
        """
        Check if this SecretRational is not equal to another.

        Args:
            other (_NadaRational): The other SecretRational to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            SecretBoolean: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        return SecretBoolean(self.value != other.value)

    def public_equals(self, other: _NadaRational) -> PublicBoolean:
        """
        Check if this SecretRational is equal to another and reveal the result.

        Args:
            other (_NadaRational): The other SecretRational to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            PublicBoolean: Result of the comparison, revealed.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        return self.value.public_equals(other.value)

    def reveal(self) -> Rational:
        """
        Reveal the SecretRational value.

        Returns:
            Rational: Revealed SecretRational value.
        """
        return Rational(self.value.reveal(), self.log_scale)

    def trunc_pr(self, arg_0: _NadaRational) -> "SecretRational":
        """
        Truncate the SecretRational value.

        Args:
            arg_0 (_NadaRational): The value to truncate by.

        Returns:
            SecretRational: Truncated SecretRational value.
        """
        return SecretRational(self.value.trunc_pr(arg_0), self.log_scale)

    def rescale_up(self, log_scale: Optional[int] = None) -> "SecretRational":
        """
        Rescale the SecretRational value upwards by a scaling factor.

        Args:
            log_scale (int, optional): The scaling factor. Defaults to RationalConfig.log_scale.

        Returns:
            SecretRational: Rescaled SecretRational value.
        """
        if log_scale is None:
            log_scale = get_log_scale()

        return SecretRational(
            self._value << UnsignedInteger(log_scale),
            self.log_scale + log_scale,
            is_scaled=True,
        )

    def rescale_down(self, log_scale: Optional[int] = None) -> "SecretRational":
        """
        Rescale the SecretRational value downwards by a scaling factor.

        Args:
            log_scale (int, optional): The scaling factor. Defaults to RationalConfig.

        Returns:
            SecretRational: Rescaled SecretRational value.
        """
        if log_scale is None:
            log_scale = get_log_scale()

        return SecretRational(
            self._value >> UnsignedInteger(log_scale),
            self.log_scale - log_scale,
            is_scaled=True,
        )

    # Non-linear functions

    def sign(self) -> "SecretRational":
        """Computes the sign value (0 is considered positive)"""

        result = sign(self)
        if not isinstance(result, SecretRational):
            raise TypeError("sign input should be of type SecretRational.")
        return result

    def abs(self) -> "SecretRational":
        """Computes the absolute value"""

        result = fxp_abs(self)
        if not isinstance(result, SecretRational):
            raise TypeError("abs input should be of type SecretRational.")
        return result

    def exp(self, iterations: int = 8) -> "SecretRational":
        """
        Approximates the exponential function using a limit approximation.

        The exponential function is approximated using the following limit:

            exp(x) = lim_{n -> ∞} (1 + x / n) ^ n

        The exponential function is computed by choosing n = 2 ** d, where d is
        set to `iterations`. The calculation is performed by computing (1 + x / n)
        once and then squaring it `d` times.

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
            SecretRational: The approximated value of the exponential function.
        """

        result = exp(self, iterations=iterations)
        if not isinstance(result, SecretRational):
            raise TypeError("exp input should be of type SecretRational.")
        return result

    def polynomial(self, coefficients: list) -> "SecretRational":
        """
        Computes a polynomial function on a value with given coefficients.

        The coefficients can be provided as a list of values.
        They should be ordered from the linear term (order 1) first, ending with the
        highest order term.
        **Note: The constant term is not included.**

        Args:
            coefficients (list): The coefficients of the polynomial, ordered by increasing degree.

        Returns:
            SecretRational: The result of the polynomial function applied to the input x.
        """

        result = polynomial(self, coefficients=coefficients)
        if not isinstance(result, SecretRational):
            raise TypeError("polynomial input should be of type SecretRational.")
        return result

    def log(
        self,
        input_in_01: bool = False,
        iterations: int = 2,
        exp_iterations: int = 8,
        order: int = 8,
    ) -> "SecretRational":
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
            exp_iterations (int, optional): Number of iterations for the limit approximation of
                exp. Defaults to 8.
            order (int, optional): Number of polynomial terms used (order of
                Householder approximation). Defaults to 8.

        Returns:
            SecretRational: The approximate value of the natural logarithm.
        """

        result = log(
            self,
            input_in_01=input_in_01,
            iterations=iterations,
            exp_iterations=exp_iterations,
            order=order,
        )
        if not isinstance(result, SecretRational):
            raise TypeError("log input should be of type SecretRational.")
        return result

    def reciprocal(  # pylint: disable=too-many-arguments
        self,
        all_pos: bool = False,
        initial: Optional["Rational"] = None,
        input_in_01: bool = False,
        iterations: int = 10,
        log_iters: int = 1,
        exp_iters: int = 8,
        method: str = "NR",
    ) -> "SecretRational":
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
            SecretRational: The approximate value of the reciprocal

        .. _Newton-Raphson:
            https://en.wikipedia.org/wiki/Newton%27s_method
        """

        result = reciprocal(
            self,
            all_pos=all_pos,
            initial=initial,
            input_in_01=input_in_01,
            iterations=iterations,
            log_iters=log_iters,
            exp_iters=exp_iters,
            method=method,
        )
        if not isinstance(result, SecretRational):
            raise TypeError("reciprocal input should be of type SecretRational.")
        return result

    def inv_sqrt(
        self,
        initial: Optional["SecretRational"] = None,
        iterations: int = 5,
        method: str = "NR",
    ) -> "SecretRational":
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
            SecretRational: The approximate value of the inv_sqrt.

        .. _Newton-Raphson:
            https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
        """

        result = inv_sqrt(self, initial=initial, iterations=iterations, method=method)
        if not isinstance(result, SecretRational):
            raise TypeError("inv_sqrt input should be of type SecretRational.")
        return result

    def sqrt(
        self,
        initial: Optional["SecretRational"] = None,
        iterations: int = 5,
        method: str = "NR",
    ) -> "SecretRational":
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
            initial (Union[SecretRational, None], optional): sets the initial value for the
                inverse square root Newton-Raphson iterations. By default, this will be set
                to allow convergence over a fairly large domain. Defaults to None.
            iterations (int, optional):  determines the number of Newton-Raphson iterations to run.
                Defaults to 5.
            method (str, optional): method used to compute sqrt. Defaults to "NR".

        Returns:
            SecretRational: The approximate value of the sqrt.

        .. _Newton-Raphson:
            https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
        """

        result = sqrt(self, initial=initial, iterations=iterations, method=method)
        if not isinstance(result, SecretRational):
            raise TypeError("sqrt input should be of type SecretRational.")
        return result

    # Trigonometry

    def cossin(self, iterations: int = 10) -> Tuple["SecretRational", "SecretRational"]:
        r"""Computes cosine and sine through e^(i * input) where i is the imaginary unit
        through the formula:

        .. math::
            Re\{e^{i * input}\}, Im\{e^{i * input}\} = \cos(input), \sin(input)

        Args:
            iterations (int, optional): determines the number of iterations to run. Defaults to 10.

        Returns:
            Tuple[SecretRational, SecretRational]:
                A tuple where the first element is cos and the second element is the sin.
        """

        result = cossin(self, iterations=iterations)
        if not isinstance(result, SecretRational):
            raise TypeError("cossin input should be of type SecretRational.")
        return result

    def cos(self, iterations: int = 10) -> "SecretRational":
        r"""Computes the cosine of the input using cos(x) = Re{exp(i * x)}.

        Note: unstable outside [-30, 30]

        Args:
            iterations (int, optional): determines the number of iterations to run. Defaults to 10.

        Returns:
            SecretRational: The approximate value of the cosine.
        """

        result = cos(self, iterations=iterations)
        if not isinstance(result, SecretRational):
            raise TypeError("cos input should be of type SecretRational.")
        return result

    def sin(self, iterations: int = 10) -> "SecretRational":
        r"""Computes the sine of the input using sin(x) = Im{exp(i * x)}.

        Note: unstable outside [-30, 30]

        Args:
            iterations (int, optional): determines the number of iterations to run. Defaults to 10.

        Returns:
            SecretRational: The approximate value of the sine.
        """

        result = sin(self, iterations=iterations)
        if not isinstance(result, SecretRational):
            raise TypeError("sin input should be of type SecretRational.")
        return result

    def tan(self, iterations: int = 10) -> "SecretRational":
        r"""Computes the tan of the input using tan(x) = sin(x) / cos(x).

        Note: unstable outside [-30, 30]

        Args:
            iterations (int, optional): determines the number of iterations to run. Defaults to 10.

        Returns:
            SecretRational: The approximate value of the tan.
        """

        result = tan(self, iterations=iterations)
        if not isinstance(result, SecretRational):
            raise TypeError("tan input should be of type SecretRational.")
        return result

    # Activation functions

    def tanh(
        self, chebyshev_terms: int = 32, method: str = "reciprocal"
    ) -> "SecretRational":
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
            SecretRational: The tanh evaluation.

        Raises:
            ValueError: Raised if method type is not supported.
        """

        result = tanh(self, chebyshev_terms=chebyshev_terms, method=method)
        if not isinstance(result, SecretRational):
            raise TypeError("tanh input should be of type SecretRational.")
        return result

    def sigmoid(
        self, chebyshev_terms: int = 32, method: str = "reciprocal"
    ) -> "SecretRational":
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
                on section 5.3 based on the Motzkin’s polynomial preprocessing technique.
                It uses the identity:

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
            method (str, optional): method used to compute sigmoid function.
                Defaults to "reciprocal".

        Returns:
            SecretRational: The sigmoid evaluation.

        Raises:
            ValueError: Raised if method type is not supported.
        """

        result = sigmoid(self, chebyshev_terms=chebyshev_terms, method=method)
        if not isinstance(result, SecretRational):
            raise TypeError("sigmoid input should be of type SecretRational.")
        return result

    def gelu(
        self, method: str = "tanh", tanh_method: str = "reciprocal"
    ) -> "SecretRational":
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
            SecretRational: The gelu evaluation.

        Raises:
            ValueError: Raised if method type is not supported.
        """

        result = gelu(self, method=method, tanh_method=tanh_method)
        if not isinstance(result, SecretRational):
            raise TypeError("gelu input should be of type SecretRational.")
        return result

    def silu(
        self,
        method_sigmoid: str = "reciprocal",
    ) -> "SecretRational":
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
                on section 5.3 based on the Motzkin’s polynomial preprocessing technique.
                It uses the identity:

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
            SecretRational: The sigmoid evaluation.

        Raises:
            ValueError: Raised if sigmoid method type is not supported.
        """

        result = silu(self, method_sigmoid=method_sigmoid)
        if not isinstance(result, SecretRational):
            raise TypeError("silu input should be of type SecretRational.")
        return result


def secret_rational(
    name: str, party: Party, log_scale: Optional[int] = None, is_scaled: bool = True
) -> SecretRational:
    """
    Creates a SecretRational from a variable in the Nillion network.

    Args:
        name (str): Name of variable in Nillion network.
        party (Party): Name of party that provided variable.
        log_scale (int, optional): Quantization scaling factor. Defaults to None.
        is_scaled (bool, optional): Flag that indicates whether provided value has already been
            scaled by log_scale factor. Defaults to True.

    Returns:
        SecretRational: Instantiated SecretRational object.
    """
    value = SecretInteger(Input(name=name, party=party))
    return SecretRational(value, log_scale, is_scaled)


def public_rational(
    name: str, party: Party, log_scale: Optional[int] = None, is_scaled: bool = True
) -> Rational:
    """
    Creates a Rational from a variable in the Nillion network.

    Args:
        name (str): Name of variable in Nillion network.
        party (Party): Name of party that provided variable.
        log_scale (int, optional): Quantization scaling factor. Defaults to None.
        is_scaled (bool, optional): Flag that indicates whether provided value has already been
            scaled by log_scale factor. Defaults to True.

    Returns:
        Rational: Instantiated Rational object.
    """
    value = PublicInteger(Input(name=name, party=party))
    return Rational(value, log_scale, is_scaled)


def rational(
    value: Union[int, float, np.floating],
    log_scale: Optional[int] = None,
    is_scaled: bool = False,
) -> Rational:
    """
    Creates a Rational from a number variable.

    Args:
        value (Union[int, float, np.floating]): Provided input value.
        log_scale (int, optional): Quantization scaling factor. Defaults to default log_scale.
        is_scaled (bool, optional): Flag that indicates whether provided value has already been
            scaled by log_scale factor. Defaults to True.

    Returns:
        Rational: Instantiated Rational object.
    """
    if value == 0:  # no use in rescaling 0
        return Rational(Integer(0), is_scaled=True)

    if log_scale is None:
        log_scale = get_log_scale()

    if isinstance(value, np.floating):
        value = value.item()
    if isinstance(value, int):
        return Rational(Integer(value), log_scale=log_scale, is_scaled=is_scaled)
    if isinstance(value, float):
        assert (
            is_scaled is False
        ), "Got a value of type `float` with `is_scaled` set to True. This should never occur"
        quantized = round(value * (1 << log_scale))
        return Rational(Integer(quantized), is_scaled=True)
    raise TypeError(f"Cannot instantiate Rational from type `{type(value)}`.")


class _MetaRationalConfig(type):
    """Rational config metaclass that defines classproperties"""

    _log_scale: int
    _default_log_scale: int

    @property
    def default_log_scale(cls) -> int:
        """
        Getter method.

        Returns:
            int: Default log scale.
        """
        return cls._default_log_scale

    @property
    def log_scale(cls) -> int:
        """
        Getter method.

        Returns:
            int: Log scale.
        """
        return cls._log_scale

    @log_scale.setter
    def log_scale(cls, new_log_scale: int) -> None:
        """
        Setter method.

        Args:
            new_log_scale (int): New log scale value to reset old value with.
        """
        if new_log_scale <= 4:
            warnings.warn(
                f"Provided log scale `{str(new_log_scale)}` is very low."
                " Expected a value higher than 4."
                " Using a low quantization scale can lead to poor quantization of rational values"
                " and thus poor performance & unexpected results."
            )
        if new_log_scale >= 64:
            warnings.warn(
                f"Provided log scale `{str(new_log_scale)}` is very high."
                " Expected a value lower than 64."
                " Using a high quantization scale can lead to overflows & unexpected results."
            )

        cls._log_scale = new_log_scale


# pylint:disable=too-few-public-methods
class _RationalConfig(metaclass=_MetaRationalConfig):
    """Rational config data class"""

    _default_log_scale: int = 16
    _log_scale: int = _default_log_scale


def set_log_scale(new_log_scale: int) -> None:
    """
    Sets the default Rational log scaling factor to a new value.
    Note that this value is the LOG scale and will be used as a base-2 exponent
    during quantization.

    Args:
        new_log_scale (int): New log scaling factor.
    """
    if not isinstance(new_log_scale, int):
        raise TypeError(
            f"Cannot set log scale to type `{type(new_log_scale)}`. Expected `int`."
        )
    _RationalConfig.log_scale = new_log_scale


def get_log_scale() -> int:
    """
    Gets the Rational log scaling factor
    Note that this value is the LOG scale and is used as a base-2 exponent during quantization.

    Returns:
        int: Current log scale in use.
    """
    return _RationalConfig.log_scale


def reset_log_scale() -> None:
    """Resets the Rational log scaling factor to the original default value"""
    _RationalConfig.log_scale = _RationalConfig.default_log_scale


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


def sign(x: _NadaRational) -> _NadaRational:
    """Computes the sign value (0 is considered positive)"""

    ltz_cond = x < rational(0)
    ltz = ltz_cond.if_else(rational(1), rational(0))

    return rational(1) - ltz - ltz


def fxp_abs(x: _NadaRational) -> _NadaRational:
    """Computes the absolute value of a rational"""
    return x * sign(x)


def exp(x: _NadaRational, iterations: int = 8) -> _NadaRational:
    """
    Approximates the exponential function using a limit approximation.
    """

    iters_na = UnsignedInteger(iterations)

    result = rational(1) + (x >> iters_na)
    for _ in range(iterations):
        result = result**2
    return result


def polynomial(x: _NadaRational, coefficients: List[Rational]) -> _NadaRational:
    """
    Computes a polynomial function on a value with given coefficients.

    The coefficients can be provided as a list of values.
    They should be ordered from the linear term (order 1) first, ending with the highest order term.
    **Note: The constant term is not included.**
    """
    result = coefficients[0] * x

    for power, coeff in enumerate(coefficients[1:], start=2):
        result += coeff * (x**power)

    return result


def log(
    x: _NadaRational,
    input_in_01: bool = False,
    iterations: int = 2,
    exp_iterations: int = 8,
    order: int = 8,
) -> _NadaRational:
    """
    Approximates the natural logarithm using 8th order modified Householder iterations.
    """

    if input_in_01:
        return log(
            x * rational(100),
            iterations=iterations,
            exp_iterations=exp_iterations,
            order=order,
        ) - rational(4.605170)

    # Initialization to a decent estimate (found by qualitative inspection):
    #                ln(x) = x/120 - 20exp(-2x - 1.0) + 3.0
    term1 = x * rational(1 / 120.0)
    term2 = exp(-x - x - rational(1), iterations=exp_iterations) * rational(20)
    y = term1 - term2 + rational(3.0)

    # 8th order Householder iterations
    for _ in range(iterations):
        h = rational(1) - x * exp(-y, iterations=exp_iterations)
        y -= polynomial(h, [rational(1 / (i + 1)) for i in range(order)])
    return y


def reciprocal(  # pylint: disable=too-many-arguments
    x: _NadaRational,
    all_pos: bool = False,
    initial: Optional[Rational] = None,
    input_in_01: bool = False,
    iterations: int = 10,
    log_iters: int = 1,
    exp_iters: int = 8,
    method: str = "NR",
) -> _NadaRational:
    """
    Approximates the reciprocal of a number through two possible methods: Newton-Raphson
    and log.
    """
    if input_in_01:
        rec = reciprocal(
            x * rational(64),
            method=method,
            all_pos=True,
            initial=initial,
            iterations=iterations,
        ) * rational(64)
        return rec

    if not all_pos:
        sgn = sign(x)
        pos = sgn * x
        return sgn * reciprocal(
            pos, method=method, all_pos=True, initial=initial, iterations=iterations
        )

    if method == "NR":
        if initial is None:
            # Initialization to a decent estimate (found by qualitative inspection):
            #                1/x = 3exp(1 - 2x) + 0.003
            result = rational(3) * exp(
                rational(1) - x - x, iterations=exp_iters
            ) + rational(0.003)
        else:
            result = initial
        for _ in range(iterations):
            result = result + result - result * result * x
        return result
    if method == "log":
        return exp(-log(x, iterations=log_iters), iterations=exp_iters)
    raise ValueError(f"Invalid method {method} given for reciprocal function")


def inv_sqrt(
    x: _NadaRational,
    initial: Optional[Union[_NadaRational, None]] = None,
    iterations: int = 5,
    method: str = "NR",
) -> _NadaRational:
    """
    Computes the inverse square root of the input using the Newton-Raphson method.
    """

    if method == "NR":
        if initial is None:
            # Initialization to a decent estimate (found by qualitative inspection):
            #                 exp(- x/2 - 0.2) * 2.2 + 0.2 - x/1024
            y = exp(-(x >> UnsignedInteger(1)) - rational(0.2)) * rational(
                2.2
            ) + rational(0.2)
            y -= x >> UnsignedInteger(10)  # div by 1024
        else:
            y = initial

        # Newton Raphson iterations for inverse square root
        for _ in range(iterations):
            y = (y * (rational(3) - x * y * y)) >> UnsignedInteger(1)
        return y
    raise ValueError(f"Invalid method {method} given for inv_sqrt function")


def sqrt(
    x: _NadaRational,
    initial: Union[_NadaRational, None] = None,
    iterations: int = 5,
    method: str = "NR",
) -> _NadaRational:
    """
    Computes the square root of the input by computing its inverse square root using
    the Newton-Raphson method and multiplying by the input.
    """

    if method == "NR":
        return inv_sqrt(x, initial=initial, iterations=iterations, method=method) * x

    raise ValueError(f"Invalid method {method} given for sqrt function")


# Trigonometry


def _eix(x: _NadaRational, iterations: int = 10) -> Tuple[_NadaRational, _NadaRational]:
    r"""Computes e^(i * x) where i is the imaginary unit through the formula:

    .. math::
        Re\{e^{i * x}\}, Im\{e^{i * x}\} = \cos(x), \sin(x)

    Args:
        x (Union[Rational, SecretRational]): the input value.
        iterations (int, optional): determines the number of iterations to run. Defaults to 10.

    Returns:
        Tuple[Union[Rational, SecretRational], Union[Rational, SecretRational]]:
            A tuple where the first element is cos and the second element is the sin.
    """

    one = rational(1)
    im = x >> UnsignedInteger(iterations)

    # First iteration uses knowledge that `re` is public and = 1
    re = one - im * im
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
    x: _NadaRational, iterations: int = 10
) -> Tuple[_NadaRational, _NadaRational]:
    r"""
    Computes cosine and sine through e^(i * x) where i is the imaginary unit.
    """
    return _eix(x, iterations=iterations)


def cos(x: _NadaRational, iterations: int = 10) -> _NadaRational:
    r"""
    Computes the cosine of x using cos(x) = Re{exp(i * x)}.
    """
    return cossin(x, iterations=iterations)[0]


def sin(x: _NadaRational, iterations: int = 10) -> _NadaRational:
    r"""
    Computes the sine of x using sin(x) = Im{exp(i * x)}.
    """
    return cossin(x, iterations=iterations)[1]


def tan(x: _NadaRational, iterations: int = 10) -> _NadaRational:
    r"""
    Computes the tan of x using tan(x) = sin(x) / cos(x).
    """
    c, s = cossin(x, iterations=iterations)
    return s * reciprocal(c)


# Activation functions


@functools.lru_cache(maxsize=10)
def chebyshev_series(func, width, terms):
    """
    Computes Chebyshev coefficients.
    """
    n_range = np.arange(start=0, stop=terms, dtype=float)
    x = width * np.cos((n_range + 0.5) * np.pi / terms)
    y = func(x)
    cos_term = np.cos(np.outer(n_range, n_range + 0.5) * np.pi / terms)
    coeffs = (2 / terms) * np.sum(y * cos_term, axis=1)
    return coeffs


def tanh(
    x: _NadaRational, chebyshev_terms: int = 32, method: str = "reciprocal"
) -> _NadaRational:
    """
    Computes the hyperbolic tangent function.
    """

    if method == "reciprocal":
        return sigmoid(x + x, method=method) * rational(2) - rational(1)
    if method == "chebyshev":
        coeffs = chebyshev_series(np.tanh, 1, chebyshev_terms)[1::2]
        # transform np.array of float to na.array of rationals
        coeffs = np.vectorize(rational)(coeffs)
        out = _chebyshev_polynomials(x, chebyshev_terms).transpose() @ coeffs
        # truncate outside [-maxval, maxval]
        return _hardtanh(x, out)
    if method == "motzkin":
        # Using approximation from "BOLT: Privacy-Preserving, Accurate and Efficient
        # Inference for Transformers"
        # section 5.3 based on the Motzkin’s polynomial preprocessing technique.

        # ltz is used for absolute value of x and to compute sign (used to generate result).
        # We don't use 'abs' and 'sign' functions to avoid computing ltz twice.
        # sign = 1 - 2 * ltz, where ltz = (x < rational(0)).if_else(rational(1), rational(0))
        sgn = rational(1) - rational(2) * (x < rational(0)).if_else(
            rational(1), rational(0)
        )
        # absolute value
        abs_x = x * sgn

        # Motzkin’s polynomial preprocessing
        t0 = rational(-4.259314087994767)
        t1 = rational(18.86353816972803)
        t2 = rational(-36.42402897526823)
        t3 = rational(-0.013232131886235352)
        t4 = rational(-3.3289339650097993)
        t5 = rational(-0.0024920889620412097)
        tanh_p0 = (abs_x + t0) * abs_x + t1
        tanh_p1 = (tanh_p0 + abs_x + t2) * tanh_p0 * t3 * abs_x + t4 * abs_x + t5

        return (abs_x > rational(2.855)).if_else(sgn, sgn * tanh_p1)
    raise ValueError(f"Unrecognized method {method} for tanh")


### Auxiliary functions for tanh


def _chebyshev_polynomials(x: _NadaRational, terms: int) -> np.ndarray:
    """Evaluates odd degree Chebyshev polynomials at x.

    Chebyshev Polynomials of the first kind are defined as:

    .. math::
        P_0(x) = 1, P_1(x) = x, P_n(x) = 2 P_{n - 1}(x) - P_{n-2}(x)

    Args:
        x (Union["Rational", "SecretRational"]): input at which polynomials are evaluated
        terms (int): highest degree of Chebyshev polynomials.
                        Must be even and at least 6.
    Returns:
        NadaArray of polynomials evaluated at x of shape `(terms, *x)`.

    Raises:
        ValueError: Raised if 'terrms' is odd and < 6.
    """
    if terms % 2 != 0 or terms < 6:
        raise ValueError("Chebyshev terms must be even and >= 6")

    # Initiate base polynomials
    # P_0
    # polynomials = np.array([x])
    # y = rational(4) * x * x - rational(2)
    # z = y - rational(1)
    # # P_1
    # polynomials = np.append(polynomials, z * x)

    # # Generate remaining Chebyshev polynomials using the recurrence relation
    # for k in range(2, terms // 2):
    #     next_polynomial = y * polynomials[k - 1] - polynomials[k - 2]
    #     polynomials = np.append(polynomials, next_polynomial)

    # return polynomials

    polynomials = [x]
    y = rational(4) * x * x - rational(2)
    z = y - rational(1)
    # P_1
    polynomials.append(z * x)

    # Generate remaining Chebyshev polynomials using the recurrence relation
    for k in range(2, terms // 2):
        next_polynomial = y * polynomials[k - 1] - polynomials[k - 2]
        polynomials.append(next_polynomial)

    return np.array(polynomials)


def _hardtanh(
    x: _NadaRational,
    output: _NadaRational,
    abs_const: _NadaRational = rational(1),
    abs_range: _NadaRational = rational(1),
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
        x (Union[Rational, SecretRational]): the input value of the Tanh.
        output (Union[Rational, SecretRational]): the output value of the approximation of Tanh.
        abs_const (Union[Rational, SecretRational]): constant value to which |Tanh(x)| converges.
            Defaults to 1.
        abs_range (Union[Rational, SecretRational]): absolute value of the range. Defaults to 1.

    Returns:
        Union[Rational, SecretRational]: HardTanh output.
    """
    # absolute value
    sgn = sign(x)
    abs_x = x * sgn
    # chekc if inside [-abs_range, abs_range] interval
    ineight_cond = abs_x < abs_range
    result = ineight_cond.if_else(output, abs_const * sgn)

    return result


### End of auxiliary functions for tanh


def sigmoid(
    x: _NadaRational, chebyshev_terms: int = 32, method: str = "reciprocal"
) -> _NadaRational:
    """
    Computes the sigmoid function.
    """
    if method == "chebyshev":
        tanh_approx = tanh(
            x >> UnsignedInteger(1), method=method, chebyshev_terms=chebyshev_terms
        )
        return (tanh_approx >> UnsignedInteger(1)) + rational(0.5)
    if method == "motzkin":
        tanh_approx = tanh(
            x >> UnsignedInteger(1), method=method, chebyshev_terms=chebyshev_terms
        )
        return (tanh_approx >> UnsignedInteger(1)) + rational(0.5)
    if method == "reciprocal":
        # ltz is used for absolute value of x and to generate 'result'.
        # We don't use 'abs' function to avoid computing ltz twice.
        ltz_cond = x < rational(0)
        ltz = ltz_cond.if_else(rational(1), rational(0))
        # compute absolute value of x
        sgn = rational(1) - rational(2) * ltz
        pos_x = x * sgn

        denominator = exp(-pos_x) + rational(1)
        pos_output = reciprocal(
            denominator, all_pos=True, initial=rational(0.75), iterations=3, exp_iters=9
        )

        # result is equivalent to (1 - ltz).if_else(pos_output, 1 - pos_output)
        result = pos_output + ltz - rational(2) * pos_output * ltz
        return result
    raise ValueError(f"Unrecognized method {method} for sigmoid")


def gelu(
    x: _NadaRational, method: str = "tanh", tanh_method: str = "reciprocal"
) -> _NadaRational:
    """
    Computes the gelu function.
    """

    if method == "tanh":
        # Using common approximation:
        #       x/2 * (1 + tanh(0.797884560 * ( x + 0.04471 * x ** 3 ) ) )
        # where 0.797884560 ~= sqrt(2/pi).
        val = rational(0.797884560) * (x + rational(0.044715) * x**3)
        return (x * (rational(1) + tanh(val, method=tanh_method))) >> UnsignedInteger(1)
    if method == "motzkin":
        # Using approximation from "BOLT: Privacy-Preserving, Accurate and Efficient
        # Inference for Transformers"
        # section 5.2 based on the Motzkin’s polynomial preprocessing technique.

        # ltz is used for absolute value of x and to compute relu.
        # We don't use 'abs' and '_relu' functions to avoid computing ltz twice.
        ltz = (x < rational(0)).if_else(rational(1), rational(0))
        # absolute value
        sgn = rational(1) - rational(2) * ltz
        abs_x = x * sgn
        # relu
        relu = x * (rational(1) - ltz)

        # Motzkin’s polynomial preprocessing
        g0 = rational(0.14439048359960427)
        g1 = rational(-0.7077117131613893)
        g2 = rational(4.5702822654246535)
        g3 = rational(-8.15444702051307)
        g4 = rational(16.382265425072532)
        gelu_p0 = (g0 * abs_x + g1) * abs_x + g2
        gelu_p1 = (gelu_p0 + g0 * abs_x + g3) * gelu_p0 + g4 + (x >> UnsignedInteger(1))

        return (abs_x > rational(2.7)).if_else(relu, gelu_p1)
    raise ValueError(f"Unrecognized method {method} for gelu")


def silu(
    x: _NadaRational,
    method_sigmoid: str = "reciprocal",
) -> _NadaRational:
    """
    Computes the gelu function
    """
    return x * sigmoid(x, method=method_sigmoid)

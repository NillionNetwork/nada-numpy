"""Additional special data types"""

# pylint:disable=too-many-lines

import warnings
from typing import Optional, Union

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


class Rational:
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


class SecretRational:
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
            log_scale (int, optional): The scaling factor. Defaults to RationalConfig.log_scale.

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
    Note that this value is the LOG scale and will be used as a base-2 exponent during quantization.

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

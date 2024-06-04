"""Additional data types"""

import numpy as np

import nada_dsl as dsl

from nada_dsl import (
    Input,
    Party,
    UnsignedInteger,
    Integer,
    NadaType,
    SecretInteger,
    SecretUnsignedInteger,
    PublicInteger,
    PublicUnsignedInteger,
)


from typing import Union


_Number = Union[float, int, np.floating]
_NadaInteger = Union[
    Integer,
    PublicInteger,
]
_NadaLike = Union[int, "Rational", "SecretRational", NadaType]

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

    def __init__(self, value):
        super().__init__(value.inner)

    def if_else(
        self: dsl.SecretBoolean,
        arg_0: _NadaType | "SecretRational" | "Rational",
        arg_1: _NadaType | "SecretRational" | "Rational",
    ) -> Union[SecretInteger, SecretUnsignedInteger]:
        first_arg = arg_0
        second_arg = arg_1
        if isinstance(arg_0, (SecretRational, Rational)) and isinstance(
            arg_1, (SecretRational, Rational)
        ):
            # Both are SecretRational or Rational objects
            if arg_0.log_scale != arg_1.log_scale:
                raise ValueError("Cannot output values with different scales.")
            first_arg = arg_0.value
            second_arg = arg_1.value
        elif isinstance(arg_0, (Rational, SecretRational)) or isinstance(
            arg_1, (Rational, SecretRational)
        ):
            # Both are SecretRational or Rational objects
            raise TypeError(f"Invalid operation: {self}.IfElse({arg_0}, {arg_1})")

        result = super().if_else(first_arg, second_arg)

        if isinstance(arg_0, (SecretRational, Rational)):
            # If we have a SecretBoolean, the return type will be SecretInteger, thus promoted to SecretRational
            return SecretRational.from_parts(result, arg_0.log_scale)
        else:
            return result


class PublicBoolean(dsl.PublicBoolean):

    def __init__(self, value):
        super().__init__(value.inner)

    def if_else(
        self: dsl.SecretBoolean,
        arg_0: _NadaType | "SecretRational" | "Rational",
        arg_1: _NadaType | "SecretRational" | "Rational",
    ) -> Union[SecretInteger, SecretUnsignedInteger]:
        first_arg = arg_0
        second_arg = arg_1
        if isinstance(arg_0, (SecretRational, Rational)) and isinstance(
            arg_1, (SecretRational, Rational)
        ):
            # Both are SecretRational or Rational objects
            if arg_0.log_scale != arg_1.log_scale:
                raise ValueError("Cannot output values with different scales.")
            first_arg = arg_0.value
            second_arg = arg_1.value
        elif isinstance(arg_0, (Rational, SecretRational)) or isinstance(
            arg_1, (Rational, SecretRational)
        ):
            # Both are SecretRational or Rational objects but of different type
            raise TypeError(f"Invalid operation: {self}.IfElse({arg_0}, {arg_1})")

        result = super().if_else(first_arg, second_arg)

        if isinstance(arg_0, (SecretRational, Rational)):
            # If we have a SecretBoolean, the return type will be SecretInteger, thus promoted to SecretRational
            return Rational.from_parts(result, arg_0.log_scale)
        else:
            return result


class RationalConfig(object):

    LOG_SCALE = 16

    @classmethod
    def set_rational_scale(cls, scale: int) -> None:
        """Sets the global scaling factor.

        Args:
            scale (int): New scaling factor in number of bits.
        """
        cls.LOG_SCALE = scale


class Rational:
    """Wrapper class to store scaled Integer values representing a fixed-point number."""

    def __init__(
        self,
        value: _Number,
        log_scale=RationalConfig.LOG_SCALE,
        is_scaled: bool = False,
    ) -> None:
        """Initializes wrapper around Integer object.

        Args:
            value (_Number): The value to be represented as a Rational.
            log_scale (int): Quantization scaling factor. Defaults to RationalConfig.LOG_SCALE.
            is_scaled (bool, optional): Flag that represents whether the value is already scaled.
                Defaults to False.

        Raises:
            TypeError: Raised when a value of an incompatible type is passed.
        """
        if not isinstance(value, (_Number, _NadaInteger)):
            raise TypeError("Cannot instantiate Rational from type `%s`." % type(value))

        if not is_scaled:
            value *= 1 << log_scale
            value = round(value)

        self._log_scale = log_scale
        if isinstance(value, _NadaInteger):
            self._value = value
        else:
            self._value = Integer(value)

    @property
    def log_scale(self) -> int:
        """Getter for the logarithmic scale value.

        Returns:
            int: Logarithmic scale value.
        """
        return self._log_scale

    @property
    def value(self) -> Integer:
        """Getter for the underlying Integer value.

        Returns:
            Integer: The Integer value.
        """
        return self._value

    @classmethod
    def from_parts(cls, value: Integer, scale: int) -> "Rational":
        """Creates a Rational from an Integer value and scale.

        Args:
            value (Integer): Integer value to convert.
            scale (int): Quantization scaling factor.

        Returns:
            Rational: Instantiated wrapper around number.
        """
        return Rational(value, scale, is_scaled=True)

    def add(
        self, other: _NadaRational, unchecked: bool = False
    ) -> Union["Rational", "SecretRational"]:
        """Add two rational numbers.

        Args:
            other (_NadaRational): Other rational number to add.
            unchecked (bool, optional): Flag to disable scale checking. Defaults to False.

        Returns:
            Union[Rational, SecretRational]: Result of the addition.

        Raises:
            TypeError: If the other value is of an incompatible type.
        """
        if not isinstance(other, (Rational, SecretRational)):
            raise TypeError(
                f"Operation + not allowed between types {type(self)} + {type(other)}"
            )

        if not unchecked and self.log_scale != other.log_scale:
            raise ValueError("Cannot add values with different scales.")

        if isinstance(other, SecretRational):
            return SecretRational.from_parts(other.value + self.value, self.log_scale)
        else:
            return Rational.from_parts(self.value + other.value, self.log_scale)

    def __add__(self, other: _NadaRational) -> Union["Rational", "SecretRational"]:
        return self.add(other)

    def __iadd__(self, other: _NadaRational) -> Union["Rational", "SecretRational"]:
        return self.add(other)

    def sub(
        self, other: _NadaRational, unchecked: bool = False
    ) -> Union["Rational", "SecretRational"]:
        """Subtract two rational numbers.

        Args:
            other (_NadaRational): Other rational number to subtract.

        Returns:
            Union[Rational, SecretRational]: Result of the subtraction.
            unchecked (bool, optional): Flag to disable scale checking. Defaults to False.

        Raises:
            TypeError: If the other value is of an incompatible type.
        """
        if not isinstance(other, (Rational, SecretRational)):
            raise TypeError(
                f"Operation - not allowed between types {type(self)} - {type(other)}"
            )

        if not unchecked and self.log_scale != other.log_scale:
            raise ValueError("Cannot substract values with different scales.")

        if isinstance(other, SecretRational):
            return SecretRational.from_parts(self.value - other.value, self.log_scale)
        else:
            return Rational.from_parts(self.value - other.value, self.log_scale)

    def __sub__(self, other: _NadaRational) -> Union["Rational", "SecretRational"]:
        return self.sub(other)

    def __isub__(self, other: _NadaLike) -> Union["Rational", "SecretRational"]:
        return self.sub(other)

    def mul_no_rescale(
        self, other: _NadaRational, unchecked: bool = False
    ) -> Union["Rational", "SecretRational"]:
        """Multiply two rational numbers.

        WARNING: This function does not rescale by default. Use `mul` to multiply and rescale.

        Args:
            other (_NadaRational): Other rational number to multiply.
            unchecked (bool, optional): Flag to disable scale checking. Defaults to False.

        Returns:
            Union[Rational, SecretRational]: Result of the multiplication.

        Raises:
            TypeError: If the other value is of an incompatible type.
        """
        if not isinstance(other, (Rational, SecretRational)):
            raise TypeError(
                f"Operation * not allowed between types {type(self)} * {type(other)}"
            )

        if not unchecked and self.log_scale != other.log_scale:
            raise ValueError("Cannot multiply values with different scales.")

        if isinstance(other, SecretRational):
            return SecretRational.from_parts(
                self.value * other.value, self.log_scale + other.log_scale
            )
        else:
            return Rational.from_parts(
                self.value * other.value, self.log_scale + other.log_scale
            )

    def mul(
        self, other: _NadaRational, unchecked: bool = False
    ) -> Union["Rational", "SecretRational"]:
        """Multiply two rational numbers and rescale the result.

        Args:
            other (_NadaRational): Other rational number to multiply.
            unchecked (bool, optional): Flag to disable scale checking. Defaults to False.

        Returns:
            Union[Rational, SecretRational]: Result of the multiplication.
        """
        c = self.mul_no_rescale(other, unchecked=unchecked)
        d = c.rescale_down()
        return d

    def __mul__(self, other: _NadaLike) -> Union["Rational", "SecretRational"]:
        return self.mul(other)

    def __imul__(self, other: _NadaLike) -> Union["Rational", "SecretRational"]:
        return self.mul(other)

    def divide_no_rescale(
        self, other: _NadaRational, unchecked: bool = False
    ) -> Union["Rational", "SecretRational"]:
        """Divide two rational numbers.

        Args:
            other (_NadaRational): Other rational number to divide by.

        Returns:
            Union[Rational, SecretRational]: Result of the division.

        Raises:
            TypeError: If the other value is of an incompatible type.
        """
        if not isinstance(other, (Rational, SecretRational)):
            raise TypeError(
                f"Operation / not allowed between types {type(self)} / {type(other)}"
            )

        if (
            not unchecked
            and self.log_scale != other.log_scale + RationalConfig.LOG_SCALE
        ):
            raise ValueError(
                f"Cannot divide values where scale is: {self.log_scale} / {other.log_scale}. Required scale: {self.log_scale}  / {other.log_scale + RationalConfig.LOG_SCALE}"
            )

        if isinstance(other, SecretRational):
            return SecretRational.from_parts(
                self.value / other.value, self.log_scale - other.log_scale
            )
        else:
            return Rational.from_parts(
                self.value / other.value, self.log_scale - other.log_scale
            )

    def divide(self, other: _NadaRational) -> Union["Rational", "SecretRational"]:
        """Divide two rational numbers and rescale the result.

        Args:
            other (_NadaRational): Other rational number to divide by.

        Returns:
            Union[Rational, SecretRational]: Result of the division.
        """
        a = self.rescale_up()
        c = a.divide_no_rescale(other)
        return c

    def __truediv__(self, other: _NadaRational) -> Union["Rational", "SecretRational"]:
        return self.divide(other)

    def __rtruediv__(self, other: _NadaRational) -> Union["Rational", "SecretRational"]:
        return self.divide(other)

    def __itruediv__(self, other: _NadaRational) -> Union["Rational", "SecretRational"]:
        return self.divide(other)

    def __pow__(self, other: int) -> "Rational":
        """Raise a rational number to an integer power using binary exponentiation.

        Args:
            other (int): The exponent.

        Returns:
            Rational: Result of the power operation.

        Raises:
            TypeError: If the exponent is not an integer.
        """
        if not isinstance(other, int):
            raise TypeError(
                "Cannot raise Rational to a power of type `%s`" % type(other)
            )

        result = Rational(1, self.log_scale)

        if other == 0:
            return result  # Any number to the power of 0 is 1

        base = self

        exponent = abs(other)
        while exponent > 0:
            if exponent % 2 == 1:
                result = result * base
            base *= base
            exponent //= 2

        if other < 0:
            return Rational.from_parts(1 / result.value, result.log_scale)

        return result

    def __neg__(self) -> "SecretRational":
        """Negate the SecretRational value.

        Returns:
            SecretRational: Negated SecretRational value.
        """
        return Rational.from_parts(self.value * Integer(-1), self.log_scale)

    def __lt__(self, other: _NadaRational) -> SecretBoolean:
        """Check if this SecretRational is less than another.

        Args:
            other (_NadaRational): The other SecretRational to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            SecretBoolean: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        return self.value < other.value

    def __gt__(self, other: _NadaRational) -> SecretBoolean:
        """Check if this SecretRational is greater than another.

        Args:
            other (_NadaRational): The other SecretRational to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            SecretBoolean: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        return self.value > other.value

    def __le__(self, other: _NadaRational) -> SecretBoolean:
        """Check if this SecretRational is less than or equal to another.

        Args:
            other (_NadaRational): The other SecretRational to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            SecretBoolean: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        return self.value <= other.value

    def __ge__(self, other: _NadaRational) -> SecretBoolean:
        """Check if this SecretRational is greater than or equal to another.

        Args:
            other (_NadaRational): The other SecretRational to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            SecretBoolean: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        return self.value >= other.value

    def __eq__(self, other: _NadaRational) -> SecretBoolean:
        """Check if this SecretRational is equal to another.

        Args:
            other (_NadaRational): The other SecretRational to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            SecretBoolean: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        return self.value == other.value

    def rescale_up(self, log_scale: int = RationalConfig.LOG_SCALE):
        """Rescale the value in the upward direction by a scaling factor (default RationalConfig.LOG_SCALE).

        This is equivalent to multiplying the value by `2**(log_scale)`.

        Args:
            log_scale (int): Scaling factor to rescale the value.

        Returns:
            SecretRational: Rescaled SecretRational value.
        """
        try:
            return Rational.from_parts(
                self._value << UnsignedInteger(log_scale), self.log_scale + log_scale
            )
        except TypeError:
            return Rational.from_parts(
                self._value * Integer(1 << log_scale), self.log_scale + log_scale
            )

    def rescale_down(self, log_scale: int = RationalConfig.LOG_SCALE):
        """Rescale the value in the downward direction by a scaling factor (default RationalConfig.LOG_SCALE).

        This is equivalent to dividing the value by `2**(log_scale)`.

        Args:
            log_scale (int): Scaling factor to rescale the value.

        Returns:
            SecretRational: Rescaled SecretRational value.
        """
        try:
            return Rational.from_parts(
                self._value >> UnsignedInteger(log_scale), self.log_scale - log_scale
            )
        except TypeError:
            return Rational.from_parts(
                self._value / Integer(1 << log_scale), self.log_scale - log_scale
            )


class SecretRational:
    """Wrapper class to store scaled SecretInteger values representing a fixed-point number."""

    def __init__(
        self,
        name: str = None,
        party: Party = None,
        value: SecretInteger = None,
        log_scale=RationalConfig.LOG_SCALE,
    ) -> None:
        """Initializes wrapper around SecretInteger object. The object should come scaled up by default otherwise precision may be lost.

        Args:
            name (str, optional): Name for the SecretInteger input.
            party (Party, optional): Party associated with the SecretInteger input.
            value (SecretInteger, optional): SecretInteger value to be represented as SecretRational.
            log_scale (int): Quantization scaling factor. Defaults to RationalConfig.LOG_SCALE.

        Raises:
            ValueError: If neither value nor name and party are provided.
        """
        if value is not None:
            self._value = value
            self._log_scale = log_scale

        elif name is not None and party is not None:
            self._value = SecretInteger(Input(name=name, party=party))
            self._log_scale = log_scale

        else:
            raise ValueError(
                "Either value or name and party must be provided or value must be a SecretInteger."
            )

    @property
    def log_scale(self) -> int:
        """Getter for the logarithmic scale value.

        Returns:
            int: Logarithmic scale value.
        """
        return self._log_scale

    @property
    def value(self) -> SecretInteger:
        """Getter for the underlying SecretInteger value.

        Returns:
            SecretInteger: The SecretInteger value.
        """
        return self._value

    @classmethod
    def from_parts(cls, value: Integer, log_scale: int) -> "SecretRational":
        """Creates a SecretRational from an Integer value.

        Args:
            value (Integer): Integer value to convert.
            log_scale (int): Quantization scaling factor.

        Returns:
            SecretRational: Instantiated SecretRational object.
        """
        return SecretRational(value=value, log_scale=log_scale)

    def add(self, other: _NadaRational, unchecked: bool = False) -> "SecretRational":
        """Add two SecretRational numbers.

        Args:
            other (_NadaRational): The other SecretRational to add.
            unchecked (bool, optional): Flag to disable scale checking. Defaults to False.

        Raises:
            TypeError: If the other value is not a Rational or SecretRational.

        Returns:
            SecretRational: Result of the addition.
        """
        if not isinstance(other, (Rational, SecretRational)):
            raise TypeError(
                f"Operation + not allowed between types {type(self)} + {type(other)}"
            )

        if not unchecked and self.log_scale != other.log_scale:
            raise ValueError("Cannot add values with different scales.")

        return SecretRational.from_parts(self.value + other.value, self.log_scale)

    def __add__(self, other: _NadaRational) -> "SecretRational":
        """Override the addition operator for SecretRational.

        Args:
            other (_NadaRational): The other SecretRational to add.

        Returns:
            SecretRational: Result of the addition.
        """
        return self.add(other)

    def __iadd__(self, other: _NadaRational) -> "SecretRational":
        """Override the in-place addition operator for SecretRational.

        Args:
            other (_NadaRational): The other SecretRational to add.

        Returns:
            SecretRational: Result of the addition.
        """
        return self.add(other)

    def sub(self, other: _NadaRational, unchecked: bool = False) -> "SecretRational":
        """Subtract two SecretRational numbers.

        Args:
            other (_NadaRational): The other SecretRational to subtract.
            unchecked (bool, optional): Flag to disable scale checking. Defaults to False.

        Raises:
            TypeError: If the other value is not a Rational or SecretRational.

        Returns:
            SecretRational: Result of the subtraction.
        """
        if not isinstance(other, (Rational, SecretRational)):
            raise TypeError(
                f"Operation - not allowed between types {type(self)} - {type(other)}"
            )
        if not unchecked and self.log_scale != other.log_scale:
            raise ValueError("Cannot substract values with different scales.")

        return SecretRational.from_parts(self.value - other.value, self.log_scale)

    def __sub__(self, other: _NadaRational) -> "SecretRational":
        """Override the subtraction operator for SecretRational.

        Args:
            other (_NadaRational): The other SecretRational to subtract.

        Returns:
            SecretRational: Result of the subtraction.
        """
        return self.sub(other)

    def __isub__(self, other: _NadaLike) -> "SecretRational":
        """Override the in-place subtraction operator for SecretRational.

        Args:
            other (_NadaLike): The other value to subtract.

        Returns:
            SecretRational: Result of the subtraction.
        """
        return self.sub(other)

    def mul_no_rescale(
        self, other: _NadaRational, unchecked: bool = False
    ) -> "SecretRational":
        """Multiply two SecretRational numbers.

        Args:
            other (_NadaRational): The other SecretRational to multiply.
            unchecked (bool, optional): Flag to disable scale checking. Defaults to False.


        Raises:
            TypeError: If the other value is not a Rational or SecretRational.

        Returns:
            SecretRational: Result of the multiplication.
        """
        if not isinstance(other, (Rational, SecretRational)):
            raise TypeError(
                f"Operation * not allowed between types {type(self)} * {type(other)}"
            )

        if not unchecked and self.log_scale != other.log_scale:
            raise ValueError("Cannot multiply values with different scales.")

        return SecretRational.from_parts(
            self.value * other.value, self.log_scale + other.log_scale
        )

    def mul(self, other: _NadaRational, unchecked: bool = False) -> "SecretRational":
        """Multiply two SecretRational numbers and rescale the result.

        Args:
            other (_NadaRational): The other SecretRational to multiply.
            unchecked (bool, optional): Flag to disable scale checking. Defaults to False.

        Returns:
            SecretRational: Result of the multiplication, rescaled.
        """
        c = self.mul_no_rescale(other, unchecked=unchecked)
        d = c.rescale_down()
        return d

    def __mul__(self, other: _NadaLike) -> "SecretRational":
        """Override the multiplication operator for SecretRational.

        Args:
            other (_NadaLike): The other value to multiply.

        Returns:
            SecretRational: Result of the multiplication, rescaled.
        """
        return self.mul(other)

    def __imul__(self, other: _NadaLike) -> "SecretRational":
        """Override the in-place multiplication operator for SecretRational.

        Args:
            other (_NadaLike): The other value to multiply.

        Returns:
            SecretRational: Result of the multiplication, rescaled.
        """
        return self.mul(other)

    def divide_no_rescale(
        self, other: _NadaRational, unchecked: bool = False
    ) -> "SecretRational":
        """Divide two SecretRational numbers.

        Args:
            other (_NadaRational): The other SecretRational to divide by.
            unchecked (bool, optional): Flag to disable scale checking. Defaults to False.

        Raises:
            TypeError: If the other value is not a Rational or SecretRational.

        Returns:
            SecretRational: Result of the division.
        """
        if not isinstance(other, (Rational, SecretRational)):
            raise TypeError(
                f"Operation / not allowed between types {type(self)} / {type(other)}"
            )

        if (
            not unchecked
            and self.log_scale != other.log_scale + RationalConfig.LOG_SCALE
        ):
            raise ValueError(
                f"Cannot divide values where scale is: {self.log_scale} / {other.log_scale}. Required scale: {self.log_scale}  / {other.log_scale + RationalConfig.LOG_SCALE}"
            )

        return SecretRational.from_parts(
            self.value / other.value, self.log_scale - other.log_scale
        )

    def divide(self, other: _NadaRational, unchecked: bool = False) -> "SecretRational":
        """Divide two SecretRational numbers and rescale the result.

        Args:
            other (_NadaRational): The other SecretRational to divide by.
            unchecked (bool, optional): Flag to disable scale checking. Defaults to False.

        Returns:
            SecretRational: Result of the division, rescaled.
        """
        a = self.rescale_up()
        c = a.divide_no_rescale(other, unchecked=unchecked)
        return c

    def __truediv__(self, other: _NadaRational) -> "SecretRational":
        """Override the true division operator for SecretRational.

        Args:
            other (_NadaRational): The other value to divide by.

        Returns:
            SecretRational: Result of the division, rescaled.
        """
        return self.divide(other)

    def __rtruediv__(self, other: _NadaRational) -> "SecretRational":
        """Override the reflected true division operator for SecretRational.

        Args:
            other (_NadaRational): The other value to divide by.

        Returns:
            SecretRational: Result of the division, rescaled.
        """
        return self.divide(other)

    def __itruediv__(self, other: _NadaRational) -> "SecretRational":
        """Override the in-place true division operator for SecretRational.

        Args:
            other (_NadaRational): The other value to divide by.

        Returns:
            SecretRational: Result of the division, rescaled.
        """
        return self.divide(other)

    def __pow__(self, other: int) -> "SecretRational":
        """Raise a SecretRational to an integer power using binary exponentiation.

        Args:
            other (int): The exponent.

        Raises:
            TypeError: If the exponent is not an integer.

        Returns:
            SecretRational: Result of the power operation.
        """
        if not isinstance(other, int):
            raise TypeError(
                "Cannot raise SecretRational to a power of type `%s`" % type(other)
            )

        result = Rational(1, self.log_scale)

        if other == 0:
            return result  # Any number to the power of 0 is 1

        base = self

        exponent = abs(other)
        while exponent > 0:
            if exponent % 2 == 1:
                result = result * base
            base *= base
            exponent //= 2

        if other < 0:
            return SecretRational.from_parts(1 / result.value, result.log_scale)

        return result

    def __neg__(self) -> "SecretRational":
        """Negate the SecretRational value.

        Returns:
            SecretRational: Negated SecretRational value.
        """
        return SecretRational.from_parts(self.value * Integer(-1), self.log_scale)

    def __lshift__(self, other: _NadaLike) -> "SecretRational":
        """Left shift the SecretRational value.

        Args:
            other (_NadaLike): The value to left shift by.

        Returns:
            SecretRational: Left shifted SecretRational value.
        """
        return SecretRational(self.value << other, self.log_scale)

    def __rshift__(self, other: _NadaLike) -> "SecretRational":
        """Right shift the SecretRational value.

        Args:
            other (_NadaLike): The value to right shift by.

        Returns:
            SecretRational: Right shifted SecretRational value.
        """
        return SecretRational(self.value >> other, self.log_scale)

    def __lt__(self, other: _NadaRational) -> SecretBoolean:
        """Check if this SecretRational is less than another.

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
        """Check if this SecretRational is greater than another.

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
        """Check if this SecretRational is less than or equal to another.

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
        """Check if this SecretRational is greater than or equal to another.

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

    def __eq__(self, other: _NadaRational) -> SecretBoolean:
        """Check if this SecretRational is equal to another.

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

    def __ne__(self, other: _NadaRational) -> SecretBoolean:
        """Check if this SecretRational is not equal to another.

        Args:
            other (_NadaRational): The other SecretRational to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            SecretBoolean: Result of the comparison.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        return not (self.value == other.value)

    def public_equals(self, other: _NadaRational) -> PublicBoolean:
        """Check if this SecretRational is equal to another and reveal the result.

        Args:
            other (_NadaRational): The other SecretRational to compare against.

        Raises:
            ValueError: If the log scales of the two values are different.

        Returns:
            PublicBoolean: Result of the comparison, revealed.
        """
        if self.log_scale != other.log_scale:
            raise ValueError("Cannot compare values with different scales.")
        return Rational.from_parts(
            self.value.public_equals(other.value), self.log_scale
        )

    def reveal(self) -> Rational:
        """Reveal the SecretRational value.

        Returns:
            Rational: Revealed SecretRational value.
        """
        return SecretRational.from_parts(self.value.reveal(), self.log_scale)

    def trunc_pr(self, arg_0: _NadaLike) -> "SecretRational":
        """Truncate the SecretRational value.

        Args:
            arg_0 (_NadaLike): The value to truncate by.

        Returns:
            SecretRational: Truncated SecretRational value.
        """
        return SecretRational.from_parts(self.value.trunc_pr(arg_0), self.log_scale)

    def rescale_up(self, log_scale: int = RationalConfig.LOG_SCALE) -> "SecretRational":
        """Rescale the SecretRational value upwards by a scaling factor.

        Args:
            log_scale (int): The scaling factor.

        Returns:
            SecretRational: Rescaled SecretRational value.
        """
        try:
            return SecretRational.from_parts(
                self._value << UnsignedInteger(log_scale), self.log_scale + log_scale
            )
        except TypeError:
            return SecretRational.from_parts(
                self._value * Integer(1 << log_scale), self.log_scale + log_scale
            )

    def rescale_down(
        self, log_scale: int = RationalConfig.LOG_SCALE
    ) -> "SecretRational":
        """Rescale the SecretRational value downwards by a scaling factor.

        Args:
            log_scale (int): The scaling factor.

        Returns:
            SecretRational: Rescaled SecretRational value.
        """
        try:
            return SecretRational.from_parts(
                self._value >> UnsignedInteger(log_scale), self.log_scale - log_scale
            )
        except TypeError:
            return SecretRational.from_parts(
                self._value / Integer(1 << log_scale), self.log_scale - log_scale
            )

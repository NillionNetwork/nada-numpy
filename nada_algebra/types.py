"""Additional data types"""

import numpy as np

from nada_dsl import (
    UnsignedInteger,
    Integer,
    NadaType,
    SecretInteger,
    SecretUnsignedInteger,
    SecretBoolean,
    PublicBoolean,
    PublicInteger,
    PublicUnsignedInteger,
)
from typing import Any, Callable, Union


_Number = Union[float, int, np.floating]
_NadaSecretInteger = Union[SecretInteger, SecretUnsignedInteger]
_NadaInteger = Union[
    Integer,
    UnsignedInteger,
    PublicInteger,
    PublicUnsignedInteger,
]
_NadaLike = Union[int, "Rational", "SecretRational", NadaType]


class Rational:
    """Wrapper class to store scaled Integer values representing e.g. a python float"""

    def __init__(
        self, value: _NadaInteger, scale: UnsignedInteger, is_scaled: bool = True
    ) -> None:
        """Initializes wrapper around Integer object.

        Args:
            value (_NadaInteger): _NadaInteger value that represents a float.
            scale (UnsignedInteger): Quantization scaling factor.
            is_scaled (bool, optional): Flag that represents whether the value is already scaled.
                Defaults to True.

        Raises:
            TypeError: Raised when a value of an incompatible type is passed.
        """
        if not isinstance(value, _NadaInteger):
            raise TypeError("Cannot instantiate Rational from type `%s`." % type(value))

        self._scale = scale
        self._value = value if is_scaled else rescale(value, scale, "up")

    @property
    def scale(self) -> UnsignedInteger:
        """Getter method. Avoids shooting of one's foot by restricting access
        to the underlying value to read-only.

        Returns:
            UnsignedInteger: UnsignedInteger value.
        """
        return self._scale

    @property
    def value(self) -> _NadaInteger:
        """Getter method. Avoids shooting of one's foot by restricting access
        to the underlying value to read-only.

        Returns:
            _NadaInteger: Integer value.
        """
        return self._value

    @classmethod
    def from_number(
        cls, value: _Number, scale: UnsignedInteger, is_scaled: bool = False
    ) -> "Rational":
        """Converts and scales a Python-native (or light wrapper by e.g., NumPy)
        number (floating-point or integer).

        Args:
            value (Number): Python-native number.
            scale (UnsignedInteger): Quantization scaling factor.
            is_scaled (bool, optional): Flag that represents whether the value is already scaled.
                Defaults to False.

        Raises:
            ValueError: Raised when a value is passed that cannot be converted to a Rational.
            TypeError: Raised when a value of an incompatible type is passed.

        Returns:
            Rational: Instantiated wrapper around number.
        """
        if value is None:
            raise ValueError("Cannot convert `%s` to Rational." % value)

        value = value.item() if isinstance(value, np.floating) else value

        if not isinstance(value, (float, int)):
            raise TypeError("Cannot instantiate Rational from type `%s`" % type(value))

        quantized = (
            Integer(value) if is_scaled else Integer(round(value * (2**scale.value)))
        )

        return Rational(quantized, scale)

    def __add__(self, other: _NadaLike) -> Union["Rational", "SecretRational"]:
        return apply_arithmetic_op(lambda x, y: x + y, self, other)

    def __radd__(self, other: _NadaLike) -> Union["Rational", "SecretRational"]:
        return apply_arithmetic_op(lambda x, y: y + x, self, other)

    def __sub__(self, other: _NadaLike) -> Union["Rational", "SecretRational"]:
        return apply_arithmetic_op(lambda x, y: x - y, self, other)

    def __rsub__(self, other: _NadaLike) -> Union["Rational", "SecretRational"]:
        return apply_arithmetic_op(lambda x, y: y - x, self, other)

    def __mul__(self, other: _NadaLike) -> Union["Rational", "SecretRational"]:
        return apply_arithmetic_op(lambda x, y: x * y, self, other, op_rescaling="down")

    def __rmul__(self, other: _NadaLike) -> Union["Rational", "SecretRational"]:
        return apply_arithmetic_op(lambda x, y: y * x, self, other, op_rescaling="down")

    def __truediv__(self, other: _NadaLike) -> Union["Rational", "SecretRational"]:
        return apply_arithmetic_op(lambda x, y: x / y, self, other, op_rescaling="up")

    def __rtruediv__(self, other: _NadaLike) -> Union["Rational", "SecretRational"]:
        return apply_arithmetic_op(lambda x, y: y / x, self, other, op_rescaling="up")

    def __mod__(self, other: _NadaLike) -> Union["Rational", "SecretRational"]:
        return apply_arithmetic_op(lambda x, y: x % y, self, other)

    def __rmod__(self, other: _NadaLike) -> Union["Rational", "SecretRational"]:
        return apply_arithmetic_op(lambda x, y: y % x, self, other)

    def __pow__(self, other: int) -> "Rational":
        if not isinstance(other, int):
            raise TypeError(
                "Cannot raise Rational to a power of type `%s`" % type(other)
            )
        # TODO: try to group truncation if no overflow
        result = self.value
        for _ in range(other - 1):
            result = rescale(result * self.value, self.scale, "down")
        return Rational(result, self.scale)

    def __neg__(self) -> "Rational":
        if isinstance(self.value, (UnsignedInteger, PublicUnsignedInteger)):
            raise TypeError("Cannot take negative of unsigned integer")
        return Rational(self.value * Integer(-1), self.scale)

    def __lt__(self, other: _NadaLike) -> SecretBoolean:
        return apply_comparison_op(lambda x, y: x < y, self, other)

    def __gt__(self, other: _NadaLike) -> SecretBoolean:
        return apply_comparison_op(lambda x, y: x > y, self, other)

    def __le__(self, other: _NadaLike) -> SecretBoolean:
        return apply_comparison_op(lambda x, y: x <= y, self, other)

    def __ge__(self, other: _NadaLike) -> SecretBoolean:
        return apply_comparison_op(lambda x, y: x >= y, self, other)

    def __eq__(self, other: _NadaLike) -> SecretBoolean:
        return apply_comparison_op(lambda x, y: x == y, self, other)


class SecretRational:
    """Wrapper class to store scaled SecretInteger values representing e.g. a python float"""

    def __init__(
        self, value: _NadaSecretInteger, scale: UnsignedInteger, is_scaled: bool = True
    ) -> None:
        """Initializes wrapper around _NadaSecretInteger object.

        Args:
            value (_NadaSecretInteger): SecretInteger value that represents a float.
            scale (UnsignedInteger): Quantization scaling factor.
            is_scaled (bool, optional): Flag that represents whether the value is already scaled.
                Defaults to True.

        Raises:
            NotImplementedError: Raised when an incompatible dtype is passed.
        """
        if not isinstance(value, _NadaSecretInteger):
            raise NotImplementedError(
                "Cannot instantiate SecretRational from type `%s`" % type(value)
            )

        self._scale = scale
        self._value = value if is_scaled else value << self._scale

    @property
    def value(self) -> _NadaSecretInteger:
        """Getter method. Avoids shooting of one's foot by restricting access
        to the underlying value to read-only.

        Returns:
            _NadaSecretInteger: SecretInteger value that this class wraps around
        """
        return self._value

    @property
    def scale(self) -> UnsignedInteger:
        """Getter method. Avoids shooting of one's foot by restricting access
        to the underlying value to read-only.

        Returns:
            UnsignedInteger: UnsignedInteger value.
        """
        return self._scale

    def __add__(self, other: _NadaLike) -> "SecretRational":
        return apply_arithmetic_op(lambda x, y: x + y, self, other)

    def __radd__(self, other: _NadaLike) -> "SecretRational":
        return apply_arithmetic_op(lambda x, y: y + x, self, other)

    def __sub__(self, other: _NadaLike) -> "SecretRational":
        return apply_arithmetic_op(lambda x, y: x - y, self, other)

    def __rsub__(self, other: _NadaLike) -> "SecretRational":
        return apply_arithmetic_op(lambda x, y: y - x, self, other)

    def __mul__(self, other: _NadaLike) -> "SecretRational":
        return apply_arithmetic_op(lambda x, y: x * y, self, other, op_rescaling="down")

    def __rmul__(self, other: _NadaLike) -> "SecretRational":
        return apply_arithmetic_op(lambda x, y: y * x, self, other, op_rescaling="down")

    def __truediv__(self, other: _NadaLike) -> "SecretRational":
        return apply_arithmetic_op(lambda x, y: x / y, self, other, op_rescaling="up")

    def __rtruediv__(self, other: _NadaLike) -> "SecretRational":
        return apply_arithmetic_op(lambda x, y: y / x, self, other, op_rescaling="up")

    def __mod__(self, other: _NadaLike) -> "SecretRational":
        return apply_arithmetic_op(lambda x, y: x % y, self, other)

    def __rmod__(self, other: _NadaLike) -> "SecretRational":
        return apply_arithmetic_op(lambda x, y: y % x, self, other)

    def __pow__(self, other: _NadaLike) -> "SecretRational":
        if not isinstance(other, int):
            raise TypeError(
                "Cannot raise SecretRational to power of type `%s`"
                % type(other).__name__
            )
        # TODO: try to group truncation if no overflow
        result = self.value
        for _ in range(other - 1):
            result = rescale(result * self.value, self.scale, "down")
        return SecretRational(result, self.scale)

    def __neg__(self) -> "SecretRational":
        if isinstance(self.value, SecretUnsignedInteger):
            raise TypeError("Cannot take negative of unsigned integer")
        return SecretRational(self.value * Integer(-1), self.scale)

    def __lshift__(self, other: _NadaLike) -> "SecretRational":
        return SecretRational(self.value << other, self.scale)

    def __rshift__(self, other: _NadaLike) -> "SecretRational":
        return SecretRational(self.value >> other, self.scale)

    def __lt__(self, other: _NadaLike) -> SecretBoolean:
        return apply_comparison_op(lambda x, y: x < y, self, other)

    def __gt__(self, other: _NadaLike) -> SecretBoolean:
        return apply_comparison_op(lambda x, y: x > y, self, other)

    def __le__(self, other: _NadaLike) -> SecretBoolean:
        return apply_comparison_op(lambda x, y: x <= y, self, other)

    def __ge__(self, other: _NadaLike) -> SecretBoolean:
        return apply_comparison_op(lambda x, y: x >= y, self, other)

    def __eq__(self, other: _NadaLike) -> SecretBoolean:
        return apply_comparison_op(lambda x, y: x == y, self, other)

    def public_equals(self, other: _NadaLike) -> PublicBoolean:
        return apply_comparison_op(lambda x, y: x.public_equals(y), self, other)

    def reveal(self) -> PublicInteger:
        return self.value.reveal()

    def trunc_pr(self, arg_0: _NadaLike) -> "SecretRational":
        return apply_arithmetic_op(lambda x, y: x.trunc_pr(y), self, arg_0)


def rescale(value: NadaType, scale: UnsignedInteger, direction: str) -> NadaType:
    """Rescales value in specified direction.

    Args:
        value (NadaType): Unscaled value.
        scale (UnsignedInteger): Scaling value.
        direction (str): Scaling direction. Either "up" or "down".

    Raises:
        ValueError: Raised when an invalid scaling direction is passed.

    Returns:
        NadaType: Scaled value.
    """
    if direction == "up":
        # TODO: remove try block when lshift implemented for every NadaType
        try:
            return value << scale
        except:
            return value * Integer(1 << scale)
    elif direction == "down":
        # TODO: remove try block when rshift implemented for every NadaType
        try:
            return value >> scale
        except:
            return value / Integer(1 << scale)

    raise ValueError(
        'Invalid scaling direction `%s`. Expected "up" or "down"' % direction
    )


def apply_arithmetic_op(
    op: Callable[[Any, Any], Any],
    this: Union[Rational, SecretRational],
    other: _NadaLike,
    op_rescaling: str = None,
) -> Union[Rational, SecretRational]:
    """Applies arithmetic operation between this value and an other value, accounting
    for any possible rescaling.

    Args:
        op (Callable[[Any, Any], Any]): Operation to apply between self and other.
        this (Union[Rational, SecretRational]): This value.
        other (_NadaLike): Other value.
        op_rescaling (str, optional): Rescaling direction after operation has been
            applied, if necessary. Defaults to None.

    Raises:
        TypeError: Raised when an invalid scaling direction is passed.

    Returns:
        Union[Rational, SecretRational]: Operation result.
    """
    if isinstance(other, int):
        other = Integer(other)

    if isinstance(other, (Rational, SecretRational)):
        result = op(this.value, other.value)
        if op_rescaling is not None:  # rescale after op if needed
            result = rescale(result, this.scale, op_rescaling)
        if isinstance(this, SecretRational) or isinstance(other, SecretRational):
            return SecretRational(result, this.scale)
        return Rational(result, this.scale)

    elif isinstance(other, NadaType):
        if op_rescaling is None:  # rescale unscaled other if non-scaling op
            other = rescale(other, this.scale, "up")
        result = op(this.value, other)
        if isinstance(this, SecretRational) or isinstance(other, _NadaSecretInteger):
            return SecretRational(result, this.scale)
        return Rational(result, this.scale)

    raise TypeError(
        "Cannot perform operation between type `%s` and type `%s`"
        % (type(this).__name__, type(other).__name__)
    )


def apply_comparison_op(
    comparison_op: Callable[[Any, Any], Any],
    this: Union[Rational, SecretRational],
    other: _NadaLike,
) -> Union[PublicBoolean, SecretBoolean]:
    """Applies comparison operation between this value and an other value, accounting
    for any possible rescaling.

    Args:
        comparison_op (Callable[[Any, Any], Any]): Comparison operation to apply between self and other.
        this (Union[Rational, SecretRational]): This value.
        other (_NadaLike): Other value.

    Raises:
        TypeError: Raised when an invalid scaling direction is passed.

    Returns:
        Union[PublicBoolean, SecretBoolean]: Comparison operation result.
    """
    if isinstance(other, int):
        other = Integer(other)

    if isinstance(other, (Rational, SecretRational)):
        other = other.value
    elif isinstance(other, NadaType):
        other = rescale(other, this.scale, "up")
    else:
        raise TypeError(
            "Cannot perform comparison between type `%s` and type `%s`"
            % (type(this).__name__, type(other).__name__)
        )

    return comparison_op(this.value, other)

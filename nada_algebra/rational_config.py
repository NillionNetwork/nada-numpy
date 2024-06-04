"""Contains rational config logic"""

import warnings
from dataclasses import dataclass


@dataclass
class __RationalConfig:
    """Rational config data class"""

    _instance: "__RationalConfig" = None

    _default_log_scale: int = 16
    _log_scale: int = _default_log_scale

    def __new__(cls, *args, **kwargs) -> "__RationalConfig":
        """
        Ensures this class is a singleton and is instantiated only once.

        Raises:
            RuntimeError: Raised when this class is attempted to be initialized more than once.

        Returns:
            __RationalConfig: New instance of class.
        """
        if cls._instance is not None:
            raise RuntimeError(
                f"{cls.__name__} class is a singleton and has already been instantiated"
            )
        cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    @property
    def log_scale(self) -> int:
        """
        Getter method.

        Returns:
            int: Log scale.
        """
        return self._log_scale

    @property
    def default_log_scale(self) -> int:
        """
        Getter method.

        Returns:
            int: Default log scale.
        """
        return self._default_log_scale

    @log_scale.setter
    def log_scale(self, new_log_scale: int) -> None:
        """
        Setter method.

        Args:
            new_log_scale (int): New log scale value to reset old value with.
        """
        if new_log_scale <= 4:
            warnings.warn(
                "Provided log scale `%s` is very low. Expected a value higher than 4."
                " Using a low quantization scale can lead to poor quantization of rational values"
                " and thus poor performance & unexpected results." % str(new_log_scale)
            )
        if new_log_scale >= 64:
            warnings.warn(
                "Provided log scale `%s` is very high. Expected a value lower than 64."
                " Using a high quantization scale can lead to overflows & unexpected results."
                % str(new_log_scale)
            )

        self._log_scale = new_log_scale


RATIONAL_CONFIG = __RationalConfig()


def set_log_scale(new_log_scale: int) -> None:
    """
    Sets the default Rational log scaling factor to a new value.
    Note that this value is the LOG scale and will be used as a base-2 exponent during quantization.

    Args:
        new_log_scale (int): New log scaling factor.
    """
    if not isinstance(new_log_scale, int):
        raise TypeError(
            "Cannot set log scale to type `%s`. Expected `int`."
            % type(new_log_scale).__name__
        )
    RATIONAL_CONFIG.log_scale = new_log_scale


def get_log_scale() -> int:
    """
    Gets the Rational log scaling factor
    Note that this value is the LOG scale and is used as a base-2 exponent during quantization.

    Returns:
        int: Current log scale in use.
    """
    return RATIONAL_CONFIG.log_scale


def reset_log_scale() -> None:
    """Resets the Rational log scaling factor to the original default value"""
    RATIONAL_CONFIG.log_scale = RATIONAL_CONFIG.default_log_scale

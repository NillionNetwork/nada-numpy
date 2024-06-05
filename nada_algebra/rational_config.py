"""Contains rational config logic"""

import warnings

class __MetaRationalConfig(type):
    """Rational config metaclass that defines classproperties"""

    _log_scale: int
    _default_log_scale: int

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

class __RationalConfig(object, metaclass=__MetaRationalConfig):
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
            "Cannot set log scale to type `%s`. Expected `int`."
            % type(new_log_scale).__name__
        )
    __RationalConfig.log_scale = new_log_scale


def get_log_scale() -> int:
    """
    Gets the Rational log scaling factor
    Note that this value is the LOG scale and is used as a base-2 exponent during quantization.

    Returns:
        int: Current log scale in use.
    """
    return __RationalConfig.log_scale


def reset_log_scale() -> None:
    """Resets the Rational log scaling factor to the original default value"""
    __RationalConfig.log_scale = __RationalConfig.default_log_scale

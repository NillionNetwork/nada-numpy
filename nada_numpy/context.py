"""Contains useful context managers"""

from nada_numpy.types import Rational, SecretRational, _NadaRational


class UnsafeArithmeticSession:
    """
    A context manager that temporarily modifies the behavior of arithmetic operations
    for Rational and SecretRational types, disabling rescaling for multiplication and division.

    Attributes:
        mul_rational (function): Original __mul__ method of Rational.
        mul_secret_rational (function): Original __mul__ method of SecretRational.
        truediv_rational (function): Original __truediv__ method of Rational.
        truediv_secret_rational (function): Original __truediv__ method of SecretRational.
    """

    def __init__(self):
        """
        Initializes the UnsafeArithmeticSession by storing the original
        multiplication and division methods of Rational and SecretRational.
        """
        self.mul_rational = Rational.__mul__
        self.mul_secret_rational = SecretRational.__mul__

        self.truediv_rational = Rational.__truediv__
        self.truediv_secret_rational = SecretRational.__truediv__

    def __enter__(self):
        """
        Enters the context, temporarily replacing the multiplication and division
        methods of Rational and SecretRational to disable rescaling.
        """

        def mul_no_rescale_wrapper(self: Rational, other: _NadaRational):
            """
            Wrapper for Rational.__mul__ that disables rescaling.

            Args:
                self (Rational): The Rational instance.
                other (_NadaRational): The other operand.

            Returns:
                Rational: Result of the multiplication without rescaling.
            """
            return Rational.mul_no_rescale(self, other, ignore_scale=True)

        def secret_mul_no_rescale_wrapper(self: SecretRational, other: _NadaRational):
            """
            Wrapper for SecretRational.__mul__ that disables rescaling.

            Args:
                self (SecretRational): The SecretRational instance.
                other (_NadaRational): The other operand.

            Returns:
                SecretRational: Result of the multiplication without rescaling.
            """
            return SecretRational.mul_no_rescale(self, other, ignore_scale=True)

        def divide_no_rescale_wrapper(self: Rational, other: _NadaRational):
            """
            Wrapper for Rational.__truediv__ that disables rescaling.

            Args:
                self (Rational): The Rational instance.
                other (_NadaRational): The other operand.

            Returns:
                Rational: Result of the division without rescaling.
            """
            return Rational.divide_no_rescale(self, other, ignore_scale=True)

        def secret_divide_no_rescale_wrapper(
            self: SecretRational, other: _NadaRational
        ):
            """
            Wrapper for SecretRational.__truediv__ that disables rescaling.

            Args:
                self (SecretRational): The SecretRational instance.
                other (_NadaRational): The other operand.

            Returns:
                SecretRational: Result of the division without rescaling.
            """
            return SecretRational.divide_no_rescale(self, other, ignore_scale=True)

        Rational.__mul__ = mul_no_rescale_wrapper
        SecretRational.__mul__ = secret_mul_no_rescale_wrapper
        Rational.__truediv__ = divide_no_rescale_wrapper
        SecretRational.__truediv__ = secret_divide_no_rescale_wrapper

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the context, restoring the original multiplication and division methods
        of Rational and SecretRational.

        Args:
            exc_type (type): Exception type if an exception occurred, else None.
            exc_val (Exception): Exception instance if an exception occurred, else None.
            exc_tb (traceback): Traceback object if an exception occurred, else None.
        """
        # Restore the original __mul__ method
        Rational.__mul__ = self.mul_rational
        SecretRational.__mul__ = self.mul_secret_rational

        # Restore the original __truediv__ method
        Rational.__truediv__ = self.truediv_rational
        SecretRational.__truediv__ = self.truediv_secret_rational

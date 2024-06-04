"""This is the __init__.py module"""

from nada_algebra.array import NadaArray
from nada_algebra.funcs import *
from nada_algebra.types import (
    Rational,
    SecretRational,
    public_rational,
    rational,
    secret_rational,
)
from nada_algebra.rational_config import get_log_scale, reset_log_scale, set_log_scale

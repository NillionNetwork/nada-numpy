"""Contains custom typing traits"""

from typing import Union

import nada_dsl as dsl

from nada_numpy.types import (PublicBoolean, Rational, SecretBoolean,
                              SecretRational)

NadaRational = Union[
    Rational,
    SecretRational,
]

NadaInteger = Union[
    dsl.Integer,
    dsl.PublicInteger,
    dsl.SecretInteger,
]

NadaUnsignedInteger = Union[
    dsl.UnsignedInteger,
    dsl.PublicUnsignedInteger,
    dsl.SecretUnsignedInteger,
]

NadaBoolean = Union[
    dsl.Boolean,
    dsl.PublicBoolean,
    dsl.SecretBoolean,
    PublicBoolean,
    SecretBoolean,
]

NadaCleartextNumber = Union[
    dsl.Integer,
    dsl.UnsignedInteger,
    Rational,
]

NadaCleartextType = Union[
    dsl.Integer,
    dsl.UnsignedInteger,
    dsl.Boolean,
    Rational,
]

AnyNadaType = Union[
    dsl.NadaType,
    Rational,
    SecretRational,
]

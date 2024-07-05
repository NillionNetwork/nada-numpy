import numpy as np
from nada_dsl import *

import nada_numpy as na

LOG_SCALE = 16
SCALE = 1 << LOG_SCALE
PRIME_64 = 18446744072637906947
PRIME_128 = 340282366920938463463374607429104828419
PRIME_256 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF98C00003
PRIME = PRIME_64


def public_modular_inverse(
    value: PublicInteger | Integer, modulo: int
) -> PublicInteger | Integer:
    """
    Calculates the modular inverse of a public value with respect to a prime modulus.

    Args:
        `value`: The value for which the modular inverse is to be calculated.
        `modulo`: The prime modulo with respect to which the modular inverse is to be calculated.

    Returns:
        The modular inverse of the value with respect to the modulo.

    Raises:
        Exception: If the input type is not a `PublicInteger` or `Integer`.
    """
    # We cannot do `value ** Integer(modulo - 2)` because the value of modulo overflows the limit of an Integer
    # We do instead: value ** (modulo - 2) == value ** ((modulo // 2) - 1) * value ** ((modulo // 2) - 1)
    # We multiply once more (value) # if modulo is odd
    mod, rem = (
        modulo // 2,
        modulo % 2,
    )  # Unless it is prime 2, it is going to be odd, but we check in any case
    power = value ** Integer(
        mod - 1
    )  # value ** modulo = value ** (modulo // 2)  * modulo ** (modulo // 2)
    power = power * power * (value if rem else Integer(1))  # value ** mo
    return power


def private_modular_inverse(secret: SecretInteger, modulo: int) -> SecretInteger:
    """
    Calculate the modular inverse of a secret value with respect to a prime modulo.

    Args:
        secret (SecretInteger): The secret value for which the modular inverse is to be calculated.
        modulo (int): The prime modulo with respect to which the modular inverse is to be calculated.

    Returns:
        SecretInteger: The modular inverse of the secret value with respect to the modulo.
    """
    r = SecretInteger.random()

    ra = r * secret  # Masking our secret
    ra_revealed = ra.reveal()  # Revealing the masked secret

    ra_inv = public_modular_inverse(
        ra_revealed, modulo
    )  # Compute the inverse of the masked secret

    a_inv = ra_inv * r  # Unmask the secret with the random shares

    return a_inv


def nada_main():
    parties = na.parties(3)

    a = na.array([1], parties[0], "A", nada_type=SecretInteger)
    a_inv = private_modular_inverse(a[0], PRIME)

    result = a_inv * a[0]
    # A_inv = na.random([3, 3], nada_type=SecretInteger)
    # A_inv = [[SecretInteger.random() for i in range(3)] for j in range(3)]
    # outputs = na.output(A_inv, parties[2], "my_output")

    return na.output(result, parties[2], "my_output")

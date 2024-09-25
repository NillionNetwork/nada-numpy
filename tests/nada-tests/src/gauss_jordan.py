import numpy as np
from nada_dsl import *

import nada_numpy as na
from nada_numpy.array import NadaArray

# from nada_crypto import random_lu_matrix, public_modular_inverse

LOG_SCALE = 16
SCALE = 1 << LOG_SCALE
PRIME_64 = 18446744072637906947
PRIME_128 = 340282366920938463463374607429104828419
PRIME_256 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF98C00003
PRIME = PRIME_64


def public_modular_inverse(
    value: Integer | UnsignedInteger, modulo: int
) -> PublicUnsignedInteger | UnsignedInteger:
    """
    Calculates the modular inverse of a value with respect to a prime modulus.

    Args:
        `value`: The value for which the modular inverse is to be calculated.
        `modulo`: The prime modulo with respect to which the modular inverse is to be calculated.

    Returns:
        The modular inverse of the value with respect to the modulo.

    Raises:
        Exception: If the input type is not a `PublicUnsignedInteger` or `UnsignedInteger`.
    """
    return value ** UnsignedInteger(modulo - 2)


def gauss_jordan_zn(mat: na.NadaArray, modulo: int):
    """
    Perform Gauss-Jordan elimination on Z_n on a given matrix.

    Parameters:
    - `matrix` (numpy.ndarray): The input matrix to perform Gauss-Jordan elimination on.
    - `modulo` (int): The modulo representing the field `Z_n`

    Returns:
    numpy.ndarray: The reduced row echelon form of the input matrix.
    """

    # Make a copy of the matrix to avoid modifying the original
    rows = mat.inner.shape[0]
    cols = mat.inner.shape[1]

    # Forward elimination
    for i in range(rows):
        # Find pivot row
        pivot_row = i
        while pivot_row < rows and (mat[pivot_row][i] == UnsignedInteger(0)) is Boolean(
            True
        ):
            pivot_row += 1

        # Swap pivot row with current row
        mat[[i, pivot_row]] = mat[[pivot_row, i]]

        # Scale pivot row to have leading 1
        diagonal_element = mat[i][i]
        pivot_inv = public_modular_inverse(diagonal_element, modulo)

        mat[i] = mat[i] * pivot_inv

        # Perform row operations to eliminate entries below pivot
        for j in range(i + 1, rows):
            factor = mat[j][i]
            mat[j] -= mat[i] * factor

    # Backward elimination
    for i in range(rows - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            factor = mat[j][i]
            mat[j] -= mat[i] * factor

    return mat


def nada_main():
    parties = na.parties(3)

    A = na.array([3, 3], parties[0], "A", nada_type=SecretUnsignedInteger)

    A = A.to_public()
    A_inv = gauss_jordan_zn(A, PRIME)
    outputs = na.output(A_inv, parties[2], "my_output")

    return outputs

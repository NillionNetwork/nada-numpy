import time

import numpy as np
from nada_dsl import *

import nada_numpy as na
from nada_numpy.array import NadaArray

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


def create_random_upper_triangular_matrix(n: int) -> NadaArray:
    """
    Create a random upper triangular matrix with the specified dimensions.

    Args:
        n (int): The size of the matrix.
        party (Party): The party object representing the current party.
        prefix (str): A prefix string to be used for generating random values.

    Returns:
        np.ndarray: A NumPy array representing a random upper triangular matrix.

    """
    # return np.triu(create_random_array([n, n], party, prefix))
    return NadaArray(
        np.array(
            [
                [SecretInteger.random() if i <= j else Integer(0) for j in range(n)]
                for i in range(n)
            ]
        )
    )


def create_random_lower_triangular_matrix(n: int) -> NadaArray:
    """
    Create a random lower triangular matrix with the specified dimensions.

    Args:
        n (int): The size of the matrix.
        party (Party): The party object representing the current party.
        prefix (str): A prefix string to be used for generating random values.

    Returns:
        np.ndarray: A NumPy array representing a random lower triangular matrix.

    """
    # return np.tril(create_random_array([n, n], party, prefix))
    return NadaArray(
        np.array(
            [
                [
                    (
                        SecretInteger.random()
                        if i > j
                        else Integer(1) if i == j else Integer(0)
                    )
                    for j in range(n)
                ]
                for i in range(n)
            ]
        )
    )


def random_lu_matrix(n: int) -> NadaArray:
    """
    Generates a random LU matrix of size n x n.

    Parameters:
    - `n` (int): The size of the matrix.

    Returns:
    - `tuple`: A tuple containing the LU matrix and the determinant of the upper triangular matrix.
    """
    upper = create_random_upper_triangular_matrix(n)
    lower = create_random_lower_triangular_matrix(n)
    # detU = upper.inner.diagonal().prod()
    return lower @ upper, None


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


def matrix_inverse(matrix: np.ndarray, modulo: int):
    if matrix.shape[0] != matrix.shape[1]:
        raise Exception("Invalid input shape: Expected equal squared matrix")
    n = matrix.shape[0]
    R, detR = random_lu_matrix(n)  # n by n random matrix R with determinant detR

    # Revealing matrix RA
    RA = (R @ matrix).to_public()
    # # Concatenating RA and R
    RAR = RA.hstack(R)
    # Performing Gauss-Jordan elimination
    A_inv = gauss_jordan_zn(RAR, modulo)

    A_inv = A_inv[
        :, n:
    ]  # After elimination, the right half of the matrix is the inverse
    return A_inv


start_time = time.time()


def nada_main():
    parties = na.parties(3)

    A = na.array([10, 10], parties[0], "A", nada_type=SecretInteger)
    A_inv = matrix_inverse(A, PRIME)

    result = A @ A_inv

    return result.output(parties[2], "my_output")

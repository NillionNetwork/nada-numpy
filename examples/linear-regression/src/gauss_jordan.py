from nada_dsl import *
import numpy as np

import nada_numpy as na

from modular_inverse import public_modular_inverse, PRIME

# from nada_crypto import random_lu_matrix, public_modular_inverse


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


def nada_main():
    parties = na.parties(3)

    A = na.array([3, 3], parties[0], "A", nada_type=SecretInteger)

    A = A.reveal()
    A_inv = gauss_jordan_zn(A, PRIME)
    outputs = na.output(A_inv, parties[2], "my_output")

    return outputs
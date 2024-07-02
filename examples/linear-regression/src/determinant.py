from nada_dsl import *
import numpy as np
import nada_numpy as na
from nada_numpy.array import NadaArray


def get_minor(matrix, i, j):
    """Return the Minor matrix after removing the i-th row and j-th column"""
    row_removed = matrix[:i].vstack(matrix[i + 1 :])
    return row_removed[:, :j].hstack(row_removed[:, j + 1 :])


def determinant(A: NadaArray):
    """
    Recursively calculate the determinant of a matrix

    Parameters:
    - `A` (numpy.ndarray): The input matrix to calculate the determinant of.
    - `modulo` (int): The modulo representing the field `Z_n`

    Returns:
    int: The determinant of the input matrix.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    # Base case for 2x2 matrix
    if A.shape == (2, 2):
        return (A[0][0] * A[1][1]) - (A[0][1] * A[1][0])
    det = Integer(0)
    for c in range(A.shape[0]):
        det += Integer((-1) ** c) * A[0][c] * determinant(get_minor(A, 0, c))
    return det


def nada_main():
    parties = na.parties(3)

    X = na.array([3, 3], parties[0], "A", SecretInteger)
    X = X.reveal()
    detX = determinant(X)

    return na.output(detX, parties[2], "my_output")
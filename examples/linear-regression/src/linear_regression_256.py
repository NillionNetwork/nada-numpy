from nada_dsl import *
import numpy as np
import nada_numpy as na
from nada_numpy.array import NadaArray

# LOG_SCALE = 8
# SCALE = 1 << LOG_SCALE
PRIME_64 = 18446744072637906947
PRIME_128 = 340282366920938463463374607429104828419
PRIME_256 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF98C00003
PRIME = PRIME_256


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
    power = power * power * value if rem else Integer(1)  # value ** mo
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
    detU = upper.inner.diagonal().prod()
    return lower @ upper, private_modular_inverse(detU, PRIME)


def matrix_inverse(matrix: np.ndarray, modulo: int):
    if matrix.shape[0] != matrix.shape[1]:
        raise Exception("Invalid input shape: Expected equal squared matrix")
    n = matrix.shape[0]
    R, detR = random_lu_matrix(n)  # n by n random matrix R with determinant detR
    # Revealing matrix RA
    RA = (R @ matrix).reveal()
    # # Concatenating RA and R
    RAR = RA.hstack(R)
    # Performing Gauss-Jordan elimination
    A_inv = gauss_jordan_zn(RAR, modulo)
    A_inv = A_inv[
        :, n:
    ]  # After elimination, the right half of the matrix is the inverse
    return A_inv


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
        # pivot_row = i
        # while pivot_row < rows and (mat[pivot_row][i] == Integer(0)) is Boolean(True):
        #     pivot_row += 1

        # # Swap pivot row with current row
        # mat[[i, pivot_row]] = mat[[pivot_row, i]]

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


def linsol(A: NadaArray, b: NadaArray, modulo: int):

    if A.shape[0] != A.shape[1]:
        raise Exception("Invalid input shape: Expected equal squared matrix")
    n = A.shape[0]  # (n,n) matrix

    R, detR_inv = random_lu_matrix(
        n
    )  # (n, n) random matrix R with inverse determinant detR_inv

    # Revealing matrix RA
    RA = (R @ A).reveal()  # (n, n) revealed matrix

    # Computing Rb as a secret matrix multiplication of R and b
    Rb = R @ b  # (n, n) @ (n,) = (n,)

    # Concatenating RA and Rb
    RAR = RA.hstack(Rb.reshape(n, 1))

    # Performing Gauss-Jordan elimination
    A_invb = gauss_jordan_zn(RAR, modulo)
    A_invb = A_invb[:, n:].reshape(
        -1,
    )  # (n, 1) -> (n,)  # After elimination, the right half of the matrix is the inverse
    detRA = determinant(RA)  # Determinant of RA
    detA = detRA * detR_inv
    # raise Exception(type(A_invb), type(detA))
    adjAb = A_invb * detA
    return adjAb, detA


def linear_regression_zn(
    X: na.NadaArray, y: na.NadaArray, modulo: int, lambda_: PublicInteger
):
    """
    Calculate the linear regression of a dataset using Z_n field.

    Parameters:
    - `X` (na.NadaArray): The input features having shape (n, d).
    - `y` (na.NadaArray): The target values having shape (n,).
    - `modulo` (int): The modulo representing the field `Z_n`.

    """
    n, d = X.shape
    A = X.T @ X #+ na.identity(d) * lambda_  # (n,d) @ (d,n) = (d,d)
    # A = SCALE * SCALE + SCALE
    b = X.T.dot(y)  # (d,n) @ (n,) = (d,)
    adjAb, detA = linsol(A, b, modulo)
    detAw = adjAb
    return detAw, detA


def nada_main():
    parties = na.parties(3)

    X = na.array([3, 3], parties[0], "A", nada_type=SecretInteger)
    y = na.array([3], parties[1], "b", nada_type=SecretInteger)
    #lambda_ = na.array([1], parties[0], "lambda", nada_type=PublicInteger)
    lambda_ = Integer(0)

    (w, b) = linear_regression_zn(X, y, PRIME, lambda_)
    # A_inv = na.random([3, 3], nada_type=SecretInteger)
    # A_inv = [[SecretInteger.random() for i in range(3)] for j in range(3)]
    # outputs = na.output(A_inv, parties[2], "my_output")

    return w.output(parties[2], "w") + na.output(b, parties[2], "b")

    # After the output, w has to be divided by b
    # w = w / b
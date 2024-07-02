
import numpy as np
import nada_numpy as na
from nada_numpy.array import NadaArray
from nada_dsl import *

from modular_inverse import public_modular_inverse, private_modular_inverse, PRIME
from determinant import determinant
from gauss_jordan import gauss_jordan_zn



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
    identity = na.eye(d, nada_type=Integer)
    A = X.T @ X +  identity * lambda_  # (n,d) @ (d,n) = (d,d)
    # A = SCALE * SCALE + SCALE
    b = X.T @ y  # (d,n) @ (n,) = (d,)
    adjAb, detA = linsol(A, b, modulo)
    detAw = adjAb
    return detAw, detA


def nada_main():
    parties = na.parties(3)

    DIM = 3
    NUM_FEATURES = 3
    X = na.array([DIM, NUM_FEATURES], parties[0], "A", nada_type=SecretInteger)
    y = na.array([NUM_FEATURES], parties[1], "b", nada_type=SecretInteger)
    # lambda_ = na.array([1], parties[0], "lambda", nada_type=PublicInteger)
    lambda_ = Integer(0)
    (w, b) = linear_regression_zn(X, y, PRIME, lambda_)

    return w.output(parties[2], "w") + na.output(b, parties[2], "b")


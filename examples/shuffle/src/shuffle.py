import nada_numpy as na
import numpy as np
from nada_dsl import *
from nada_numpy import SecretRational
from numpy import vectorize


def rand_bool() -> SecretBoolean:
    """
    Generates a random boolean.
    """
    r = na.random((1,), SecretRational)[0]
    return r > na.rational(0)


def random_bit():
    """Generates a random bit as an integer"""
    b_bool = rand_bool()
    b_int = b_bool.if_else(Integer(1), Integer(0))

    return b_int


def random_kbit_number(k: int):
    """Generates a random number with k bits"""
    r = Integer(0)
    for i in range(k):
        # opt: random bit without comparison
        r += Integer(2**i) * random_bit()
    return r


def random_rational():
    """Random number between [0, 1)"""

    scale = na.get_log_scale()
    return SecretRational(random_kbit_number(scale), scale, is_scaled=True)


def one_hot_vector(dim: int):
    """Random one-hot vector: one index with value 1 and all others with value 0"""

    def to_int(b):
        """Transforms bool into int"""
        return b.if_else(Integer(1), Integer(0))

    vec_bool_to_int = vectorize(to_int)

    x = na.NadaArray(np.array([na.rational(i) for i in range(1, dim + 1)]))
    # opt: avoid truncation
    r = random_rational() * na.rational(dim)

    gt = x > r
    gt = vec_bool_to_int(gt)

    shifted = na.NadaArray(np.roll(gt, 1))
    shifted[0] = Integer(0)

    return na.NadaArray(gt) - shifted


def shuffle(x):
    """
    Shuffles a NadaArray x

    Currently, it does not support arrays with Rational and SecretRational types.
    """
    n = len(x)
    for i in range(n - 1):
        u = one_hot_vector(n - i)
        x_u = x[i:].dot(u)
        d = u * (x[i] - x_u)
        x[i] = x_u
        x[i:] = x[i:] + d

    return x


def nada_main():

    n = 12

    parties = na.parties(2)
    a = na.array([n], parties[0], "A", PublicInteger)
    b = na.array([n], parties[0], "B", SecretInteger)

    # As a function

    def bool_to_int(boolean):
        """Casting bool to int"""
        return boolean.if_else(Integer(0), Integer(1))

    def count(vec, element):
        """
        Counts the number of times element is in vec.
        """

        result = Integer(0)
        for e in vec:
            b = ~(element == e)
            int_b = bool_to_int(b)
            result += int_b

        return result

    shuffled_a = shuffle(a)
    shuffled_b = shuffle(b)

    # 1. Show shuffle works for PublicInteger and SecretInteger
    result_a = shuffled_a - shuffled_a
    result_b = shuffled_b - shuffled_b

    # 2. Randomness: show at least one element is in a different position
    # true if equal
    diff_position_bool = [a[i] == shuffled_a[i] for i in range(n)]
    # cast to int (true -> 0 and false -> 1)
    diff_position = np.array([bool_to_int(element) for element in diff_position_bool])
    # add them
    sum = diff_position.sum()
    # if all are equal => all are 0 => sum is zero
    at_least_one_diff_element = sum > Integer(0)

    # 3. Show elements are preserved:
    check = Integer(0)
    for ai in a:
        nr_ai_in_shufled_a = count(shuffled_a, ai)
        nr_ai_in_a = count(a, ai)
        check += bool_to_int(nr_ai_in_shufled_a == nr_ai_in_a)
    elements_are_preserved = check == Integer(0)

    return (
        na.output(result_a, parties[1], "my_output_a")
        + na.output(result_b, parties[1], "my_output_b")
        + na.output(at_least_one_diff_element, parties[1], "at_least_one_diff_element")
        + na.output(elements_are_preserved, parties[1], "elements_are_preserved")
    )

"""Main Nada program"""

from typing import List

import numpy as np
from nada_dsl import Integer, Output, PublicInteger, SecretInteger

import nada_numpy as na
from nada_numpy import shuffle


def bool_to_int(bool):
    """Casting bool to int"""
    return bool.if_else(Integer(0), Integer(1))


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


def nada_main() -> List[Output]:

    n = 8

    parties = na.parties(2)
    a = na.array([n], parties[0], "A", na.Rational)
    b = na.array([n], parties[0], "B", na.SecretRational)
    c = na.array([n], parties[0], "C", PublicInteger)
    d = na.array([n], parties[0], "D", SecretInteger)

    # As a function

    shuffled_a = shuffle(a)
    shuffled_b = shuffle(b)
    shuffled_c = shuffle(c)
    shuffled_d = shuffle(d)

    # 1. Show shuffle works for Rational, SecretRational, PublicInteger and SecretInteger
    result_a = shuffled_a - shuffled_a
    result_b = shuffled_b - shuffled_b
    result_c = shuffled_c - shuffled_c
    result_d = shuffled_d - shuffled_d

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

    # As a method

    shuffled_method_a = a.shuffle()
    shuffled_method_b = b.shuffle()
    shuffled_method_c = c.shuffle()
    shuffled_method_d = d.shuffle()

    # 1. Show shuffle works for Rational, SecretRational, PublicInteger and SecretInteger
    result_method_a = shuffled_method_a - shuffled_method_a
    result_method_b = shuffled_method_b - shuffled_method_b
    result_method_c = shuffled_method_c - shuffled_method_c
    result_method_d = shuffled_method_d - shuffled_method_d

    # 2. Randomness: show at least one element is in a different position
    # true if equal
    diff_position_bool_method = [a[i] == shuffled_method_a[i] for i in range(n)]
    # cast to int (true -> 0 and false -> 1)
    diff_position_method = np.array(
        [bool_to_int(element) for element in diff_position_bool_method]
    )
    # add them
    sum_method = diff_position_method.sum()
    # if all are equal => all are 0 => sum is zero
    at_least_one_diff_element_method = sum_method > Integer(0)

    # 3. Show elements are preserved:
    check = Integer(0)
    for ai in a:
        nr_ai_in_shufled_a = count(shuffled_method_a, ai)
        nr_ai_in_a = count(a, ai)
        check += bool_to_int(nr_ai_in_shufled_a == nr_ai_in_a)
    elements_are_preserved_method = check == Integer(0)

    return (
        na.output(result_a, parties[1], "my_output_a")
        + na.output(result_b, parties[1], "my_output_b")
        + na.output(result_c, parties[1], "my_output_c")
        + na.output(result_d, parties[1], "my_output_d")
        + na.output(result_method_a, parties[1], "my_output_method_a")
        + na.output(result_method_b, parties[1], "my_output_method_b")
        + na.output(result_method_c, parties[1], "my_output_method_c")
        + na.output(result_method_d, parties[1], "my_output_method_d")
        + na.output(at_least_one_diff_element, parties[1], "at_least_one_diff_element")
        + na.output(
            at_least_one_diff_element_method,
            parties[1],
            "at_least_one_diff_element_method",
        )
        + na.output(elements_are_preserved, parties[1], "elements_are_preserved")
        + na.output(
            elements_are_preserved_method, parties[1], "elements_are_preserved_method"
        )
    )

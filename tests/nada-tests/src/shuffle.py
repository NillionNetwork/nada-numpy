"""Main Nada program"""

from typing import List

from nada_dsl import Output, PublicInteger, SecretInteger

import nada_numpy as na
# Step 0: Nada Numpy is imported with this line
from nada_numpy import shuffle


def nada_main() -> List[Output]:

    parties = na.parties(2)
    a = na.array([8], parties[0], "A", na.Rational)
    b = na.array([8], parties[0], "B", na.SecretRational)
    c = na.array([8], parties[0], "C", PublicInteger)
    d = na.array([8], parties[0], "D", SecretInteger)

    # As a function

    shuffled_a = shuffle(a)
    shuffled_b = shuffle(b)
    shuffled_c = shuffle(c)
    shuffled_d = shuffle(d)

    result_a = shuffled_a - shuffled_a
    result_b = shuffled_b - shuffled_b
    result_c = shuffled_c - shuffled_c
    result_d = shuffled_d - shuffled_d

    # As a method

    shuffled_method_a = a.shuffle()
    shuffled_method_b = b.shuffle()
    shuffled_method_c = c.shuffle()
    shuffled_method_d = d.shuffle()

    result_method_a = shuffled_method_a - shuffled_method_a
    result_method_b = shuffled_method_b - shuffled_method_b
    result_method_c = shuffled_method_c - shuffled_method_c
    result_method_d = shuffled_method_d - shuffled_method_d

    return (
        na.output(result_a, parties[1], "my_output_a")
        + na.output(result_b, parties[1], "my_output_b")
        + na.output(result_c, parties[1], "my_output_c")
        + na.output(result_d, parties[1], "my_output_d")
        + na.output(result_method_a, parties[1], "my_output_method_a")
        + na.output(result_method_b, parties[1], "my_output_method_b")
        + na.output(result_method_c, parties[1], "my_output_method_c")
        + na.output(result_method_d, parties[1], "my_output_method_d")
    )

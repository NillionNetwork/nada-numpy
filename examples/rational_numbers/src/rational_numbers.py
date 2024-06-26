"""Main Nada program"""

from typing import List

from nada_dsl import Output

import nada_numpy as na


def nada_main() -> List[Output]:
    """
    Main dot product Nada program.

    Returns:
        List[Output]: List of program outputs.
    """
    # We define the number of parties
    parties = na.parties(3)

    # We use na.SecretRational to create a secret rational number for party 0
    a = na.secret_rational("my_input_0", parties[0])

    # We use na.SecretRational to create a secret rational number for party 1
    b = na.secret_rational("my_input_1", parties[1])

    # This is a compile time rational number
    c = na.rational(1.2)

    # The formula below does operations on rational numbers and returns a rational number
    # It's easy to see that (a + b - c) is both on numerator and denominator, so the end result is b
    out = ((a + b - c) * b) / (a + b - c)

    return na.output(out, parties[2], "my_output")

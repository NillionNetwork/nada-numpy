"""Broadcasting example program"""

from typing import List

from nada_dsl import Output, SecretInteger

import nada_algebra as na


def nada_main() -> List[Output]:
    """
    Main Nada program.

    Returns:
        List[Output]: List of Nada outputs.
    """
    # Step 1: We use Nada Algebra wrapper to create "Party0", "Party1" and "Party2"
    parties = na.parties(3)

    # Step 2: Party0 creates an array of dimension (3, ) with name "A"
    a = na.array([3], parties[0], "A", SecretInteger)

    # Step 3: Party1 creates an array of dimension (3, ) with name "B"
    b = na.array([3], parties[1], "B", SecretInteger)

    # Step 4: Party0 creates an array of dimension (3, ) with name "C"
    c = na.array([3], parties[0], "C", SecretInteger)

    # Step 5: Party1 creates an array of dimension (3, ) with name "D"
    d = na.array([3], parties[1], "D", SecretInteger)

    # Step 4: The result is of computing SIMD operations on top of the elements of the array
    # SIMD operations are performed on all the elements of the array.
    # The equivalent would be: for i in range(3): result += a[i] + b[i] - c[i] * d[i]
    result = a + b - c * d
    # Step 5: We can use result.output() to produce the output for Party2
    # and variable name "my_output"
    return result.output(parties[2], "my_output")

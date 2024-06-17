"""Main Nada program"""

from typing import List

from nada_dsl import Output, SecretInteger

import nada_algebra as na


def nada_main() -> List[Output]:
    """
    Main dot product Nada program.

    Returns:
        List[Output]: List of program outputs.
    """
    # Step 1: We use Nada Algebra wrapper to create "Party0", "Party1" and "Party2"
    parties = na.parties(3)

    # Step 2: Party0 creates an array of dimension (3 x 3) with name "A"
    a = na.array([3, 3], parties[0], "A", SecretInteger)

    # Step 3: Party1 creates an array of dimension (3 x 3) with name "B"
    b = na.array([3, 3], parties[1], "B", SecretInteger)

    # Step 4: The result is of computing the dot product between the two which is another (3 x 3) matrix
    result = a @ b

    # Step 5: We can use result.output() to produce
    # the output for Party2 and variable name "my_output"
    return result.output(parties[2], "my_output")

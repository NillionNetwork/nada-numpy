from nada_dsl import *

# Step 0: Nada Algebra is imported with this line
import nada_algebra as na


def nada_main():
    # Step 1: We use Nada Algebra wrapper to create "Party0", "Party1" and "Party2"
    parties = na.parties(3)

    # Step 2: Party0 creates an array of dimension (2 x 1 x 1 x 2 x 2 x 2) with name "A"
    a = na.array([2, 1, 1, 2, 2, 2], parties[0], "A", na.SecretRational)

    # Step 3: Party1 creates an array of dimension (2 x 1 x 1 x 2 x 2 x 2) with name "B"
    b = na.array([2, 1, 1, 2, 2, 2], parties[1], "B", na.SecretRational)

    # Step 4: The result is of computing the dot product between the two which is another (2 x 1 x 1 x 2 x 2 x 2) matrix
    result = a @ b

    # Step 5: We can use result.output() to produce the output for Party2 and variable name "my_output"
    return result.output(parties[1], "my_output")

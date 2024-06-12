# Dot Product Tutorial

This tutorial shows how to efficiently program a dot product in Nada using Nada Algebra. 

```python
from nada_dsl import *

# Step 0: Nada Algebra is imported with this line
import nada_algebra as na


def nada_main():
    # Step 1: We use Nada Algebra wrapper to create "Party0", "Party1" and "Party2"
    parties = na.parties(3)

    # Step 2: Party0 creates an array of dimension (3, ) with name "A"
    a = na.array([3], parties[0], "A", SecretInteger)

    # Step 3: Party1 creates an array of dimension (3, ) with name "B"
    b = na.array([3], parties[1], "B", SecretInteger)

    # Step 4: The result is of computing the dot product between the two
    result = a.dot(b)

    # Step 5: We can use result.output() to produce the output for Party2 and variable name "my_output"
    return na.output(result, parties[1], "my_output")

```

0. We import Nada algebra using `import nada_algebra as na`.
1. We create an array of parties, with our wrapper using `parties = na.parties(3)` which creates an array of parties named: `Party0`, `Party1` and `Party2`.
2. We create our input array `a` with `na.array([3], parties[0], "A")`, meaning our array will have dimension 3, `Party0` will be in charge of giving its inputs and the name of the variable is `"A"`.
3. We create our input array `b` with `na.array([3], parties[1], "B")`, meaning our array will have dimension 3, `Party1` will be in charge of giving its inputs and the name of the variable is `"B"`.
4. Then, we use the `dot` function to compute the dot product like `a.dot(b)`, which will encompass all the functionality.
5. Finally, we use Nada Algebra to produce the outputs of the array like:  `result.output(parties[2], "my_output")` establishing that the output party will be `Party2`and the name of the output variable will be `my_output`. 
# How to run the tutorial.

1. First, we need to compile the nada program running: `nada build`.
2. Then, we can test our program is running with: `nada test`. 

Inspecting `tests/dot-product.yml`, we see how the inputs for the file are two vectors of 3s: 

$$ A = (3, 3, 3), B = (3, 3, 3)$$
And we obtain:
$$A \times B = 3 \cdot 3 + 3 \cdot 3 + 3 \cdot 3 = 27$$
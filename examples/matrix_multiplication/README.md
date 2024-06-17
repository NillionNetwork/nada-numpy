# Matrix Multiplication Tutorial

This tutorial shows how to efficiently program a matrix multiplication in Nada using Nada Numpy. 

```python
from nada_dsl import *

# Step 0: Nada Numpy is imported with this line
import nada_numpy as na


def nada_main():
    # Step 1: We use Nada Numpy wrapper to create "Party0", "Party1" and "Party2"
    parties = na.parties(3)

    # Step 2: Party0 creates an array of dimension (3 x 3) with name "A"
    a = na.array([3, 3], parties[0], "A", SecretInteger)

    # Step 3: Party1 creates an array of dimension (3 x 3) with name "B"
    b = na.array([3, 3], parties[1], "B", SecretInteger)

    # Step 4: The result is of computing the dot product between the two which is another (3 x 3) matrix
    result = a @ b

    # Step 5: We can use result.output() to produce the output for Party2 and variable name "my_output"
    return result.output(parties[1], "my_output")

```

0. We import Nada numpy using `import nada_numpy as na`.
1. We create an array of parties, with our wrapper using `parties = na.parties(3)` which creates an array of parties named: `Party0`, `Party1` and `Party2`.
2. We create our input array `a` with `na.array([3], parties[0], "A")`, meaning our array will have dimension 3, `Party0` will be in charge of giving its inputs and the name of the variable is `"A"`.
3. We create our input array `b` with `na.array([3], parties[1], "B")`, meaning our array will have dimension 3, `Party1` will be in charge of giving its inputs and the name of the variable is `"B"`.
4. Then, we use the `dot` function to compute the dot product like `a.dot(b)`, which will encompass all the functionality.
5. Finally, we use Nada Numpy to produce the outputs of the array like:  `result.output(parties[2], "my_output")` establishing that the output party will be `Party2`and the name of the output variable will be `my_output`. 

# How to Run the Tutorial

1. Start by compiling the Nada program using the command:
   ```
   nada build
   ```

2. (Optional) Next, ensure that the program functions correctly by testing it with:
   ```
   nada test
   ```

3. Finally, we can call our Nada program via the Nillion python client by running: `python3 main.py`

Upon inspecting the `tests/matrix-multiplication.yml` file, you'll observe that the inputs consist of two matrices with dimensions (3 x 3):

$$
A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}, \quad B = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}
$$
And we obtain an output matrix:

$$
C = A \times B = \begin{pmatrix} 1 \cdot 1 + 2 \cdot 4 + 3 \cdot 7  & 1 \cdot 2 + 2 \cdot 5 + 3 \cdot 8  & 1 \cdot 3 + 2 \cdot 6 + 3 \cdot 9  \\ 4 \cdot 1 + 5 \cdot 4 + 6 \cdot 7  &  4 \cdot 2 + 5 \cdot 5 + 6 \cdot 8  & 4 \cdot 3 + 4 \cdot 6 + 6 \cdot 9 \\ 7 \cdot 1 + 8 \cdot 4 + 9 \cdot 7  &  7 \cdot 2 + 8 \cdot 5 + 9 \cdot 8  & 7 \cdot 3 + 8 \cdot 6 + 9 \cdot 9 \end{pmatrix} = \begin{pmatrix} 30 & 36 & 42 \\ 66 & 81 & 96 \\ 102 & 126 & 150 \end{pmatrix} $$ 
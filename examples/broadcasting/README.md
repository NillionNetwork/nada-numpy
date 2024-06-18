# Broadcasting Tutorial

This tutorial shows how to efficiently use broadcasting in Nada using Nada Algebra. 

```python
from nada_dsl import *

# Step 0: Nada Algebra is imported with this line
import nada_algebra as na


def nada_main():
    # Step 1: We use Nada Algebra wrapper to create "Party0", "Party1" and "Party2"
    parties = na.parties(3)

    # Step 2: Party0 creates an array of dimension (3, ) with name "A"
    a = na.array([3], parties[0], "A")

    # Step 3: Party1 creates an array of dimension (3, ) with name "B"
    b = na.array([3], parties[1], "B")

    # Step 4: Party0 creates an array of dimension (3, ) with name "C"
    c = na.array([3], parties[0], "C")

    # Step 5: Party1 creates an array of dimension (3, ) with name "D"
    d = na.array([3], parties[1], "D")

    # Step 4: The result is of computing SIMD operations on top of the elements of the array
    # SIMD operations are performed on all the elements of the array.
    # The equivalent would be: for i in range(3): result += a[i] + b[i] - c[i] * d[i]
    result = a + b - c * d
    # Step 5: We can use result.output() to produce the output for Party2 and variable name "my_output"
    return result.output(parties[2], "my_output")
```

0. We import Nada algebra using `import nada_algebra as na`.
1. We create an array of parties, with our wrapper using `parties = na.parties(3)` which creates an array of parties named: `Party0`, `Party1` and `Party2`.
2. We create our input array `a` with `na.array([3], parties[0], "A")`, meaning our array will have dimension 3, `Party0` will be in charge of giving its inputs and the name of the variable is `"A"`.
3. We create our input array `b` with `na.array([3], parties[1], "B")`, meaning our array will have dimension 3, `Party1` will be in charge of giving its inputs and the name of the variable is `"B"`.
4. We create our input array `c` with `na.array([3], parties[1], "C")`, meaning our array will have dimension 3, `Party0` will be in charge of giving its inputs and the name of the variable is `"C"`.
5. We create our input array `d` with `na.array([3], parties[1], "D")`, meaning our array will have dimension 3, `Party1` will be in charge of giving its inputs and the name of the variable is `"D"`.
5. Finally, we use Nada Algebra to produce the outputs of the array like:  `result.output(parties[2], "my_output")` establishing that the output party will be `Party2`and the name of the output variable will be `my_output`. 
# How to run the tutorial.

1. First, we need to compile the nada program running: `nada build`.
2. (Optional) Then, we can test our program is running with: `nada test`. 
3. Finally, we can call our Nada program via the Nillion python client by running: `python3 main.py`

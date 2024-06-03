# Rational Numbers Tutorial

This tutorial shows how to use Nada Algebra Rational datatypes to work with fixed-point numbers in Nada.

## Notions

This tutorial uses fixed point numbers as it is the only available way to use Fixed-Point numbers in Nada. The representation of a fixed point number uses integer places to represent decimal digits. Thus, every number is multiplied by a scaling factor, that we refer to as `SCALE` ($\Delta = 2^{16}$) or `LOG_SCALE` in its logarithmic notation ($log_2\Delta = 16$). In a nutshell, this means we will use 16 bits to represent decimals. 

If we want to input a variable `a = float(3.2)`, we need to first encode it. For that we will define a new variable `a'` which is going to be the scaled version. In this case, the scaling factor (to simplify) is going to by 3 bits so, $log_2\Delta = 3$ and $\Delta = 2^3 = 8$. With the following formula, we compute the encoded value:

$$ a' = round(a * \Delta) = round(a * 2^{log_2\Delta}) = 3.2 \cdot 2^3 = 3.2 \cdot 8 = 26 $$

Thus, in order to introduce a value with 3 bits of precision, we would be inputing 26 instead of 3.2.



## Example 

```python
from nada_dsl import *
import nada_algebra as na


def nada_main():
    # We define the number of parties
    parties = na.parties(3)

    # We use na.SecretRational to create a secret rational number for party 0
    a = na.SecretRational("my_input_0", parties[0]) 

    # We use na.SecretRational to create a secret rational number for party 1
    b = na.SecretRational("my_input_1", parties[1]) 

    # This is a compile time rational number
    c = na.Rational(1.2) 

    # The formula below does operations on rational numbers and returns a rational number
    # It's easy to see that (a + b - c) is both on numerator and denominator, so the end result is b
    out_0 = ((a + b - c) * b) / (a + b - c) 

    return [
        Output(out_0.value, "my_output_0", parties[2]),
    ]


```

0. We import Nada algebra using `import nada_algebra as na`.
1. We create an array of parties, with our wrapper using `parties = na.parties(3)` which creates an array of parties named: `Party0`, `Party1` and `Party2`.
2. We create our secret floating point variable `a` as `SecretRational("my_input_0", parties[0])` meaning the variable belongs to `Party0` and the name of the variable is `my_input_0`.
3. We create our secret floating point variable `b` as `SecretRational("my_input_1", parties[1])` meaning the variable belongs to `Party1` and the name of the variable is `my_input_1`.
4. Then, we operate normally with this variables, and Nada Algebra will ensure they maintain the consistency of the decimal places.
5. Finally, we produce the outputs of the array like:  `Output(out_0.value, "my_output_0", parties[2]),` establishing that the output party will be `Party2`and the name of the output variable will be `my_output`. Not the difference between Nada Algebra and classic Nada where we add `out_0`**`.value`**.

# How to Run the Tutorial

1. Start by compiling the Nada program using the command:
   ```
   nada build
   ```

2. Next, ensure that the program functions correctly by testing it with:
   ```
   nada test
   ```

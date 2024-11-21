import sys

import numpy as np
from nada_test import NadaTest, nada_test

import nada_numpy.client as na


# Functional style test
@nada_test(program="matrix_inverse")
def my_test():
    n = 10
    m = np.random.rand(n, n)
    mx = np.sum(np.abs(m), axis=1)
    np.fill_diagonal(m, mx)
    A = na.array(m * (1 << 16), "A", nada_type=int)
    print("INPUTS:", A, file=sys.stderr)
    outputs = yield A
    print(outputs, file=sys.stderr)
    for output, value in outputs.items():
        output = output.split("_")
        if output[-1] == output[-2]:
            assert value == 1, f"Expected 1 {output}, got {value}"
        else:
            assert value == 0, f"Expected 0 {output}, got {value}"

    # assert outputs["my_output"] == a + b

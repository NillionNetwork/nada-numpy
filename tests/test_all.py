import os
import subprocess

import pytest

TESTS = [
    "base",
    "new_array",
    "dot_product",
    "sum",
    "broadcasting_sum",
    "broadcasting_sub",
    "broadcasting_mul",
    "broadcasting_div",
    "broadcasting_vec",
    "hstack",
    "vstack",
    "reveal",
    "matrix_multiplication",
    "generate_array",
    "supported_operations",
    "get_item",
    "get_attr",
    "set_item",
    "gauss_jordan",
    "generate_array",
    "rational_arithmetic",
    "rational_comparison",
    "secret_rational_arithmetic",
    "secret_rational_comparison",
    "chained_rational_operations",
    "rational_array",
    "rational_scaling",
    "rational_operability",
    "rational_if_else",
    "random_array",
    "rational_advanced",
    "array_attributes",
    "functional_operations",
    "array_statistics",
    "matrix_multiplication_rational",
    "matrix_multiplication_rational_multidim",
    "dot_product_rational",
    "supported_operations_return_types",
    "logistic_regression",
    "logistic_regression_rational",
    "type_guardrails",
    "shape",
    "get_vec",
    "unsigned_matrix_inverse",
    "private_inverse",
    "fxpmath_funcs",
    "fxpmath_arrays",
    "fxpmath_methods",
    "shuffle",
    "array_comparison",
    "private_inverse",
]

EXAMPLES = [
    # FORMAT: (EXAMPLES_DIR (DIR), EXAMPLE_NAME (PY), TEST_NAME (YAML))
    ("dot_product", "dot_product", "dot_product"),
    ("matrix_multiplication", "matrix_multiplication", "matrix_multiplication"),
    ("broadcasting", "broadcasting", "broadcasting"),
    ("rational_numbers", "rational_numbers", "rational_numbers"),
    ("linear_regression", "determinant", "determinant_1"),
    ("linear_regression", "determinant", "determinant_2"),
    ("linear_regression", "determinant", "determinant_3"),
    ("linear_regression", "gauss_jordan", "gauss_jordan"),
    ("linear_regression", "matrix_inverse", "nada-test::matrix_inverse.py:my_test"),
    ("linear_regression", "linear_regression", "linear_regression"),
    ("linear_regression", "linear_regression", "linear_regression_1"),
    ("linear_regression", "linear_regression", "linear_regression_2"),
    ("linear_regression", "linear_regression", "linear_regression_256_1"),
    ("linear_regression", "linear_regression", "linear_regression_256_2"),
    ("linear_regression", "modular_inverse", "modular_inverse"),
]

TESTS = [("tests/nada-tests/", test, test) for test in TESTS] + [
    (os.path.join("examples/", test[0]), test[1], test[2]) for test in EXAMPLES
]


@pytest.fixture(params=TESTS)
def testname(request):
    return request.param


def build_nada(test_dir):
    print(test_dir)
    result = subprocess.run(
        ["nada", "build", test_dir[1]], cwd=test_dir[0], capture_output=True, text=True
    )
    err = result.stderr.lower() + result.stdout.lower()
    if result.returncode != 0 or "error" in err or "fail" in err:
        pytest.fail(f"Build {test_dir}:\n{result.stdout + result.stderr}")


def run_nada(test_dir):
    result = subprocess.run(
        ["nada", "test", test_dir[2]], cwd=test_dir[0], capture_output=True, text=True
    )

    # if "shape" in test_dir[1]:
    #     pytest.fail(f"Run {test_dir}:\n{result.stdout + result.stderr}")

    err = result.stderr.lower() + result.stdout.lower()
    if result.returncode != 0 or "error" in err or "fail" in err:
        pytest.fail(f"Run {test_dir}:\n{result.stdout + result.stderr}")


class TestSuite:

    def test_build(self, testname):
        # Build Nada Program
        build_nada(testname)

    def test_run(self, testname):
        # Build Nada Program
        run_nada(testname)


def test_client():
    import nillion_client as nillion
    import numpy as np

    import nada_numpy.client as na_client  # For use with Python Client

    parties = na_client.parties(3)

    assert parties is not None

    secrets = na_client.concat(
        [
            na_client.array(np.ones((3, 3)), "A", nillion.SecretInteger),
            na_client.array(np.ones((3, 3)), "B", nillion.SecretUnsignedInteger),
        ]
    )

    assert secrets is not None

    public_variables = na_client.concat(
        [
            na_client.array(np.zeros((4, 4)), "C", nillion.Integer),
            na_client.array(np.zeros((3, 3)), "D", nillion.UnsignedInteger),
        ]
    )

    assert public_variables is not None


def test_rational_client():
    import nillion_client as nillion

    import nada_numpy.client as na_client  # For use with Python Client

    secret_rational = na_client.secret_rational(3.2)

    assert isinstance(secret_rational, nillion.SecretInteger)

    rational = na_client.public_rational(1.7)

    assert isinstance(rational, nillion.Integer)

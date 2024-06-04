import os
import pytest
import subprocess

TESTS = [
    "base",
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
    "supported_operations",
    "rational_arithmetic",
    "rational_comparison",
    "secret_rational_arithmetic",
    "secret_rational_comparison",
    "chained_rational_operations",
    "rational_array",
    "rational_if_else",
    # Not supported yet
    # "unsigned_matrix_inverse",
    # "private_inverse"
    # "unsigned_matrix_inverse_2"
]

EXAMPLES = ["dot_product", "matrix_multiplication", "broadcasting", "rational_numbers"]

TESTS = [("tests/nada-tests/", test) for test in TESTS] + [
    (os.path.join("examples/", test), test) for test in EXAMPLES
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
        ["nada", "test", test_dir[1]], cwd=test_dir[0], capture_output=True, text=True
    )
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
    import nada_algebra.client as na_client  # For use with Python Client
    import py_nillion_client as nillion
    import numpy as np

    parties = na_client.parties(3)

    assert parties is not None

    secrets = nillion.Secrets(
        na_client.concat(
            [
                na_client.array(np.ones((3, 3)), "A"),
                na_client.array(np.ones((3, 3)), "B", nillion.SecretUnsignedInteger),
            ]
        )
    )

    assert secrets is not None

    public_variables = nillion.PublicVariables(
        na_client.concat(
            [
                na_client.array(np.zeros((4, 4)), "C", nillion.PublicVariableInteger),
                na_client.array(
                    np.zeros((3, 3)), "D", nillion.PublicVariableUnsignedInteger
                ),
            ]
        )
    )

    assert public_variables is not None


def test_rational_client():
    import nada_algebra.client as na_client  # For use with Python Client
    import py_nillion_client as nillion

    secret_rational = na_client.SecretRational(3.2)

    assert type(secret_rational) == nillion.SecretInteger

    rational = na_client.PublicRational(1.7)

    assert type(rational) == nillion.PublicVariableInteger

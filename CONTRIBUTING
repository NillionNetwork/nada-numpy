# Contributing to `nada-numpy`

Thank you for considering contributing to `nada-numpy`! There are two ways to contribute to `nada-numpy`:

1. [Open issues](#open-issues) to report bugs and typos, or to suggest new ideas.
2. [Submit a PR](#submit-a-pull-request) with a new feature or improvement.

To ensure a consistent development process, please follow the guidelines outlined below.

## Code quality and type checking

- All contributions must adhere to the project's coding standards. We enforce these standards using `pylint` for code quality and `mypy` for type checking. 
- Before submitting your contributions, ensure that your code passes both `pylint` and `mypy` checks.
- These tools are also integrated into our CI/CD pipeline, and any PR will be automatically validated against these checks.

## Development 

We recommend continuously running your code through `pylint` and `mypy` during the development process. These tools help identify potential issues early, enforce coding standards, and maintain type safety.

### Installation

1. Install [black](https://pypi.org/project/black/) and [isort](https://pycqa.github.io/isort/) for code formating
```bash
pip3 install black && isort
```
2. Fork the [repo](https://github.com/NillionNetwork/nada-numpy.git)
3. Install from source the `nada-numpy` library:
```bash
cd nada-numpy
pip3 install -e .
```

### Adding tests

The [`tests/nada-tests`](https://github.com/NillionNetwork/nada-numpy/tree/main/tests/nada-tests) folder contains the testing infrastructure for `nada_numpy`. You will need to create one or more scripts to test your functionality. You can read the [docs](https://docs.nillion.com/nada#generate-a-test-file) for more info about testing. Follow these steps for testing:

1. Add a script to [`tests/nada-tests/nada-project.toml`](https://github.com/NillionNetwork/nada-numpy/blob/main/tests/nada-tests/nada-project.toml).

2. Place your test script in [`tests/nada-tests/src/`](https://github.com/NillionNetwork/nada-numpy/blob/main/tests/nada-tests/src), where it will verify the expected behavior.

3. Generate the test file by running 
```bash
nada generate-test --test-name <TEST_NAME> <PROGRAM>
```
and placing it in [`tests/nada-tests/tests/`](https://github.com/NillionNetwork/nada-numpy/blob/main/tests/nada-tests/tests).

4. Finally, add the script to the `TESTS` array in [`tests/test_all.py`](https://github.com/NillionNetwork/nada-numpy/blob/dd112a09835c2354cbf7ecc89ad2714ca7171b20/tests/test_all.py#L6) to integrate it with the CI/CD pipeline.

## Submit a Pull Request

We actively welcome your pull requests. Please follow these steps to successfully submit a PR:

1. Fork the [repo](https://github.com/NillionNetwork/nada-numpy.git) and create your branch from `main`.
2. If you've added code that should be tested, add tests as explained [above](#adding-tests). 
3. Ensure that the test suite passes. Under [`tests/nada-tests/tests/`](https://github.com/NillionNetwork/nada-numpy/blob/main/tests/nada-tests/tests) run 
```bash
nada test <TEST_NAME>
```
4. Run from the root directory both 
```bash
black . && isort .
```
5. Ensure that your code passes both `pylint` and `mypy` checks:
```bash
poetry run pylint
poetry run mypy
```

## Hints to add new features

Below we provide some hints on how to add new features. We give two examples: adding a `NadaArray` method and adding a `Rational` method.

### New `NadaArray` method

As an example, we use the `variance` operation to describe the development flow:

1. **Integrate the variance function:**
   - In [`nada_numpy/array.py`](https://github.com/NillionNetwork/nada-numpy/blob/main/nada_numpy/array.py), integrate the `variance` method inside the `NadaArray` class as a new member function. This will allow the `variance` to be called as `array.var()`.

2. **Add a Wrapper in `nada_numpy/funcs.py`:**
   - In `nada-numpy`, functions can also be called in the form `na.var(array)`. To support this, add a wrapper in [`nada_numpy/funcs.py`](https://github.com/NillionNetwork/nada-numpy/blob/main/nada_numpy/funcs.py). You can refer to the existing functions in this file to see how they simply wrap around `array.var()` in this context.

### New `Rational` method

As an example, we use the exponential `exp` function to describe the development flow:

1. **Integrate the exp function with Rational:**
   - In [`nada_numpy/types.py`](https://github.com/NillionNetwork/nada-numpy/blob/main/nada_numpy/types.py), integrate the `exp` method inside both the `Rational` and `SecretRational` classes as a new member function. This will allow the `exp` to be called as `value.exp()`.

2. **Integrate the exp function with NadaArray:**
   - In [`nada_numpy/array.py`](https://github.com/NillionNetwork/nada-numpy/blob/main/nada_numpy/array.py), integrate the `exp` method inside the `NadaArray` class as a new member function. This will allow the `exp` to be called as `array.exp()`.

3. **Add a Wrapper in `nada_numpy/funcs.py`:**
   - In `nada-numpy`, functions can also be called in the form `na.exp(array)`. To support this, add a wrapper in [`nada_numpy/funcs.py`](https://github.com/NillionNetwork/nada-numpy/blob/main/nada_numpy/funcs.py). You can refer to the existing functions in this file to see how they simply wrap around `array.exp()` in this context.


## Open Issues

We use [GitHub issues](https://github.com/NillionNetwork/nada-numpy/issues/new/choose) to report bugs and typos, or to suggest new ideas. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue.

## License
By contributing to `nada-numpy`, you agree that your contributions will be licensed under the [LICENSE](./LICENSE) file in the root directory of this source tree.
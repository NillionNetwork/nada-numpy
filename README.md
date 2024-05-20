# Nada-Algebra

![GitHub License](https://img.shields.io/github/license/NillionNetwork/nada-algebra?style=for-the-badge)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/NillionNetwork/nada-algebra/test?style=for-the-badge)


Nada-Algebra is a Python library designed for algebraic operations on NumPy-like array objects on top of Nada DSL and Nillion Network. It provides a simple and intuitive interface for performing various algebraic computations, including dot products, element-wise operations, and stacking operations, while supporting broadcasting similar to NumPy arrays.

## Features

- **Dot Product**: Compute the dot product between two NadaArray objects.
- **Element-wise Operations**: Perform element-wise addition, subtraction, multiplication, and division with broadcasting support.
- **Stacking**: Horizontally and vertically stack arrays.
- **Custom Function Application**: Apply custom functions to each element of the array.

## Installation
### Using pip

```bash
pip install nada-algebra
```

### From Sources
You can install the nada-algebra library using Poetry:

```bash
git clone https://github.com/NillionNetwork/nada-algebra.git
pip3 install poetry
poetry install nada-algebra
```

## License

This project is licensed under the Apache2 License. See the LICENSE file for details.
# Nada-Numpy

![GitHub License](https://img.shields.io/github/license/NillionNetwork/nada-numpy?style=for-the-badge)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/NillionNetwork/nada-numpy/test?style=for-the-badge)


Nada-Numpy is a Python library designed for algebraic operations on NumPy-like array objects on top of Nada DSL and Nillion Network. It provides a simple and intuitive interface for performing various algebraic computations, including dot products, element-wise operations, and stacking operations, while supporting broadcasting similar to NumPy arrays.

## Features

### Use Numpy Array Features
- **Dot Product**: Compute the dot product between two NadaArray objects.
- **Element-wise Operations**: Perform element-wise addition, subtraction, multiplication, and division with broadcasting support.
- **Stacking**: Horizontally and vertically stack arrays.
### Use Decimal Numbers in Nada
- **Rational Number Support**: Our implementation of `Rational` and `SecretRational` allows the use of simplified implementations of decimal numbers on top of Nillion.

## Installation
### Using pip

```bash
pip install nada-numpy
```

### From Sources
You can install the nada-numpy library using Poetry:

```bash
git clone https://github.com/NillionNetwork/nada-numpy.git
pip3 install ./nada-numpy
```

## Testing

To test that the version installed works as expected, you can use poetry as follows:

```bash
poetry install
poetry run pytest
```

## License

This project is licensed under the Apache2 License. See the LICENSE file for details.
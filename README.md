# Nada-Numpy

![GitHub License](https://img.shields.io/github/license/NillionNetwork/nada-numpy?style=for-the-badge&logo=apache&logoColor=white&color=%23D22128&link=https%3A%2F%2Fgithub.com%2FNillionNetwork%2Fnada-numpy%2Fblob%2Fmain%2FLICENSE&link=https%3A%2F%2Fgithub.com%2FNillionNetwork%2Fnada-numpy%2Fblob%2Fmain%2FLICENSE)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/NillionNetwork/nada-numpy/test.yml?style=for-the-badge&logo=python&logoColor=white&link=https%3A%2F%2Fgithub.com%2FNillionNetwork%2Fnada-numpy%2Factions%2Fworkflows%2Ftest.yml&link=https%3A%2F%2Fgithub.com%2FNillionNetwork%2Fnada-numpy%2Factions%2Fworkflows%2Ftest.yml)
![GitHub Release](https://img.shields.io/github/v/release/NillionNetwork/nada-numpy?sort=date&display_name=release&style=for-the-badge&logo=dependabot&label=LATEST%20RELEASE&color=0000FE&link=https%3A%2F%2Fpypi.org%2Fproject%2Fnada-numpy&link=https%3A%2F%2Fpypi.org%2Fproject%2Fnada-numpy)

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
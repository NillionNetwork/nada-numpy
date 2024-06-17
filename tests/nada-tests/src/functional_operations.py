import pytest
from nada_dsl import *

import nada_numpy as na


def nada_main():
    parties = na.parties(2)

    a = na.array([3], parties[0], "A", SecretInteger)
    b = na.array([3], parties[0], "B", na.SecretRational)

    # Test all for a native NadaType
    _ = na.sum(a)
    _ = na.compress(a, [True, True, False], axis=0)
    _ = na.copy(a)
    _ = na.cumprod(a, axis=0)
    _ = na.cumsum(a, axis=0)
    _ = na.diagonal(a.reshape(1, 3))
    _ = na.prod(a)
    _ = na.put(a, 2, Integer(20))
    _ = na.ravel(a)
    _ = na.repeat(a, 12)
    _ = na.reshape(a, (1, 3))
    _ = na.resize(a, (1, 3))
    _ = na.squeeze(a.reshape(1, 3))
    _ = na.swapaxes(a.reshape(1, 3), 1, 0)
    _ = na.take(a, 1)
    _ = na.trace(a)
    _ = na.transpose(a)
    _ = na.pad(a, 2)
    _ = na.pad(a, 2, mode="constant", constant_values=Integer(1))
    _ = na.pad(a, 2, mode="edge")
    _ = na.pad(a, 2, mode="reflect")
    _ = na.pad(a, 2, mode="symmetric")
    _ = na.pad(a, 2, mode="wrap")
    with pytest.raises(TypeError):
        na.pad(a, 2, constant_values=na.rational(1))
    _ = na.pad(b, 2)
    _ = na.pad(b, 2, mode="constant", constant_values=na.rational(1))
    _ = na.pad(b, 2, mode="edge")
    _ = na.pad(b, 2, mode="reflect")
    _ = na.pad(b, 2, mode="symmetric")
    _ = na.pad(b, 2, mode="wrap")
    with pytest.raises(TypeError):
        na.pad(b, 2, constant_values=Integer(1))
    _ = na.split(a, (1, 2))

    pyfunc_out_1 = na.frompyfunc(lambda x: x + Integer(1), 1, 1)(a)
    assert isinstance(pyfunc_out_1, na.NadaArray), type(pyfunc_out_1).__name__

    with pytest.raises(TypeError):

        class Counter:
            count = 0

        def mixed_types():  # generates alternating integers & rationals
            Counter.count += 1
            if Counter.count % 2 == 0:
                return Integer(1)
            return na.rational(1)

        na.frompyfunc(mixed_types, 1, 1)(a)

    with pytest.raises(TypeError):

        class Counter:
            count = 0

        def mixed_types():  # generates alternating integers & rationals
            Counter.count += 1
            if Counter.count % 2 == 0:
                return Integer(1)
            return na.rational(1)

        na.vectorize(mixed_types)(a)

    pyfunc_out_2, pyfunc_out_3 = na.frompyfunc(
        lambda x: (x + Integer(1), x + Integer(2)), 1, 2
    )(a)
    assert isinstance(pyfunc_out_2, na.NadaArray), type(pyfunc_out_2).__name__
    assert isinstance(pyfunc_out_3, na.NadaArray), type(pyfunc_out_3).__name__

    pyfunc_out_4 = na.frompyfunc(lambda x, y: x + y, 2, 1)(a, a)
    assert isinstance(pyfunc_out_4, na.NadaArray), type(pyfunc_out_4).__name__

    vectorize_out_1 = na.vectorize(lambda x: x + Integer(1))(a)
    assert isinstance(vectorize_out_1, na.NadaArray), type(vectorize_out_1).__name__

    vectorize_out_2, vectorize_out_3 = na.vectorize(
        lambda x: (x + Integer(1), x + Integer(2))
    )(a)
    assert isinstance(vectorize_out_2, na.NadaArray), type(vectorize_out_2).__name__
    assert isinstance(vectorize_out_3, na.NadaArray), type(vectorize_out_3).__name__

    vectorize_out_4 = na.vectorize(lambda x, y: x + y)(a, a)
    assert isinstance(vectorize_out_4, na.NadaArray), type(vectorize_out_4).__name__

    # Test all for a Rational type
    _ = na.sum(b)
    _ = na.compress(b, [True, True, False], axis=0)
    _ = na.copy(b)
    _ = na.cumprod(b, axis=0)
    _ = na.cumsum(b, axis=0)
    _ = na.diagonal(b.reshape(1, 3))
    _ = na.prod(b)
    _ = na.put(b, 2, na.rational(20, is_scaled=True))
    _ = na.ravel(b)
    _ = na.repeat(b, 12)
    _ = na.reshape(b, (1, 3))
    _ = na.resize(b, (1, 3))
    _ = na.squeeze(b.reshape(1, 3))
    _ = na.swapaxes(b.reshape(1, 3), 1, 0)
    _ = na.take(b, 1)
    _ = na.trace(b)
    _ = na.transpose(b)
    _ = na.pad(b, 2)
    _ = na.pad(b, 2, mode="edge")
    _ = na.pad(b, 2, mode="reflect")
    _ = na.pad(b, 2, mode="symmetric")
    _ = na.pad(b, 2, mode="wrap")
    _ = na.split(b, (1, 2))

    pyfunc_out_5 = na.frompyfunc(lambda x: x + na.rational(1), 1, 1)(b)
    assert isinstance(pyfunc_out_5, na.NadaArray), type(pyfunc_out_4).__name__

    pyfunc_out_6, pyfunc_out_7 = na.frompyfunc(
        lambda x: (x + na.rational(1), x + na.rational(2)), 1, 2
    )(b)
    assert isinstance(pyfunc_out_6, na.NadaArray), type(pyfunc_out_6).__name__
    assert isinstance(pyfunc_out_7, na.NadaArray), type(pyfunc_out_7).__name__

    pyfunc_out_8 = na.frompyfunc(lambda x, y: x + y, 2, 1)(b, b)
    assert isinstance(pyfunc_out_8, na.NadaArray), type(pyfunc_out_8).__name__

    vectorize_out_5 = na.vectorize(lambda x: x + na.rational(1))(b)
    assert isinstance(vectorize_out_5, na.NadaArray), type(pyfunc_out_4).__name__

    vectorize_out_6, vectorize_out_7 = na.vectorize(
        lambda x: (x + na.rational(1), x + na.rational(2))
    )(b)
    assert isinstance(vectorize_out_6, na.NadaArray), type(vectorize_out_6).__name__
    assert isinstance(vectorize_out_7, na.NadaArray), type(vectorize_out_7).__name__

    vectorize_out_8 = na.vectorize(lambda x, y: x + y)(b, b)
    assert isinstance(vectorize_out_8, na.NadaArray), type(vectorize_out_8).__name__

    # Generative functions
    _ = na.eye(3, nada_type=na.Rational)
    _ = na.eye(3, nada_type=Integer)
    _ = na.arange(3, nada_type=na.Rational)
    _ = na.arange(3, nada_type=UnsignedInteger)
    _ = na.linspace(1, 4, 2, nada_type=na.Rational)
    _ = na.linspace(1, 4, 2, nada_type=Integer)

    return a.output(parties[1], "my_output_A") + b.output(parties[1], "my_output_B")

from nada_dsl import *
import nada_algebra as na


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
    _ = na.pad(a, 2, mode="edge")
    _ = na.pad(a, 2, mode="reflect")
    _ = na.pad(a, 2, mode="symmetric")
    _ = na.pad(a, 2, mode="wrap")
    _ = na.split(a, (1, 2))

    # Test all for a Rational type
    _ = na.sum(b)
    _ = na.compress(b, [True, True, False], axis=0)
    _ = na.copy(b)
    _ = na.cumprod(b, axis=0)
    _ = na.cumsum(b, axis=0)
    _ = na.diagonal(b.reshape(1, 3))
    _ = na.prod(b)
    _ = na.put(b, 2, Integer(20))
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

    # Generative functions
    _ = na.eye(3, nada_type=na.Rational)
    _ = na.eye(3, nada_type=Integer)
    _ = na.arange(3, nada_type=na.Rational)
    _ = na.arange(3, nada_type=UnsignedInteger)
    _ = na.linspace(1, 4, 2, nada_type=na.Rational)
    _ = na.linspace(1, 4, 2, nada_type=Integer)

    return a.output(parties[1], "my_output_A") + b.output(parties[1], "my_output_B")

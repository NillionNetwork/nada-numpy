import pytest
from nada_dsl import *
import nada_algebra as na


def nada_main():
    parties = na.parties(1)

    a = na.array([3, 3], parties[0], "A", SecretInteger)

    assert isinstance(a.data, memoryview)
    assert a.flags["WRITEABLE"]
    assert isinstance(na.NadaArray(a.flat), na.NadaArray)
    assert a.itemsize == 8
    assert a.nbytes == 72
    assert a.ndim == 2
    assert a.strides == (24, 8)

    with pytest.raises(AttributeError):
        a.argsort()

    a = a.compress([True, True, False], axis=0)
    a = a.copy()
    a = a.cumprod(axis=0)
    a = a.cumsum(axis=1)
    a = a.diagonal()
    a = a.item(0)

    b = na.array([3, 3], parties[0], "B", SecretInteger)
    b = b.flatten()
    b = b.prod()

    c = na.array([3, 3], parties[0], "C", SecretInteger)
    _ = c.put(3, Integer(20))
    c = c.ravel()
    c = c.reshape((3, 3))
    c = c.sum(axis=1)
    c = c.take(1)

    d = na.array([1], parties[0], "D", SecretInteger)
    d = d.repeat(12)
    d.resize((4, 3))
    twelve = Integer(d.size)
    d = d.squeeze()
    d = d.swapaxes(1, 0)
    d = d.T
    d = (d + twelve).item(0)

    e = na.array([3, 3], parties[0], "E", SecretInteger)
    five = Integer(sum(e.shape))
    e = e.transpose()
    e = e.trace()
    e = e + five

    f = na.array([1], parties[0], "F", SecretInteger)
    f.fill(Integer(40))
    f.itemset(0, f.item(0) + Integer(2))
    f = f.tolist()[0]

    return [
        Output(a, "out_0", parties[0]),
        Output(b, "out_1", parties[0]),
        Output(c, "out_2", parties[0]),
        Output(d, "out_3", parties[0]),
        Output(e, "out_4", parties[0]),
        Output(f, "out_5", parties[0]),
    ]

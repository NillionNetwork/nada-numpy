import pytest
from nada_dsl import *
import nada_algebra as na
import numpy as np


def check_array(array):
    assert isinstance(array, na.NadaArray), f"{array} is not a NadaArray: {type(array)}"
    assert isinstance(
        array.inner, np.ndarray
    ), f"{array.inner} is not a numpy array: {type(array.inner)}"


def nada_main():
    parties = na.parties(2)

    a = na.array([3, 3], parties[0], "A", SecretInteger)

    assert isinstance(a.data, memoryview)
    assert a.flags["WRITEABLE"]
    check_array(a)
    assert a.itemsize == 8
    assert a.nbytes == 72
    assert a.ndim == 2
    assert a.strides == (24, 8)

    with pytest.raises(AttributeError):
        a.argsort()

    a = a.compress([True, True, False], axis=0)
    check_array(a)
    a = a.copy()
    check_array(a)
    a = a.cumprod(axis=0)
    check_array(a)
    a = a.cumsum(axis=1)
    check_array(a)
    a = a.diagonal()
    check_array(a)
    a = a.item(0)  # Not array

    b = na.array([3, 3], parties[0], "B", SecretInteger)
    check_array(b)
    b = b.flatten()
    check_array(b)
    b = b.prod() # Not an array

    c = na.array([3, 3], parties[0], "C", SecretInteger)
    _ = c.put(3, Integer(20))
    check_array(c)
    c = c.ravel()
    check_array(c)
    c = c.reshape((3, 3))
    check_array(c)
    c = c.sum(axis=1)
    check_array(c)
    c = c.take(1) # Not an array

    d = na.array([1], parties[0], "D", SecretInteger)
    check_array(d)
    d = d.repeat(12)
    check_array(d)
    d.resize((4, 3))
    check_array(d)
    twelve = Integer(d.size)
    d = d.squeeze()
    check_array(d)
    d = d.swapaxes(1, 0)
    check_array(d)
    d = d.T
    check_array(d)
    d = (d + twelve).item(0)  # Not an array

    e = na.array([3, 3], parties[0], "E", SecretInteger)
    check_array(e)
    five = Integer(sum(e.shape))
    e = e.transpose()
    check_array(e)
    e = e.trace() # Not an array
    e = e + five # Not an array either

    f = na.array([1], parties[0], "F", SecretInteger)
    check_array(f)
    f.fill(Integer(40))
    check_array(f)
    f.itemset(0, f.item(0) + Integer(2))
    check_array(f)
    assert isinstance(f.tolist(), list)
    f = f.tolist()[0]  # Not an array

    return (
        na.output(a, parties[1], "out_0")
        + na.output(b, parties[1], "out_1")
        + na.output(c, parties[1], "out_2")
        + na.output(d, parties[1], "out_3")
        + na.output(e, parties[1], "out_4")
        + na.output(f, parties[1], "out_5")
    )

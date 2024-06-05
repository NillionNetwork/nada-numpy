import pytest
from nada_dsl import *
import nada_algebra as na


def nada_main():
    parties = na.parties(3)

    a = na.secret_rational("A", parties[0])
    b = na.secret_rational("B", parties[1])
    c = na.secret_rational("C", parties[2])
    d = SecretInteger(Input("D", parties[0]))
    e = SecretUnsignedInteger(Input("E", parties[0]))

    out_0 = (a > b).if_else(a, b)
    assert isinstance(out_0, na.SecretRational), type(out_0).__name__
    out_1 = (a >= b).if_else(a, b)
    assert isinstance(out_1, na.SecretRational), type(out_1).__name__
    out_2 = (a < b).if_else(a, b)
    assert isinstance(out_2, na.SecretRational), type(out_2).__name__
    out_3 = (a <= b).if_else(a, b)
    assert isinstance(out_3, na.SecretRational), type(out_3).__name__

    out_4 = (a > na.rational(1)).if_else(d, Integer(1))
    assert isinstance(out_4, SecretInteger), type(out_4).__name__
    out_5 = (a >= na.rational(1)).if_else(d, Integer(1))
    assert isinstance(out_5, SecretInteger), type(out_5).__name__
    out_6 = (a < na.rational(1)).if_else(d, Integer(1))
    assert isinstance(out_6, SecretInteger), type(out_6).__name__
    out_7 = (a <= na.rational(1)).if_else(d, Integer(1))
    assert isinstance(out_7, SecretInteger), type(out_7).__name__

    out_8 = (na.rational(0) > na.rational(1)).if_else(na.rational(1), na.rational(2))
    assert isinstance(out_8, na.Rational), type(out_8).__name__
    out_9 = (na.rational(0) >= na.rational(1)).if_else(na.rational(2), na.rational(1))
    assert isinstance(out_9, na.Rational), type(out_9).__name__
    out_10 = (na.rational(0) < na.rational(1)).if_else(Integer(1), Integer(2))
    assert isinstance(out_10, PublicInteger), type(out_10).__name__
    out_11 = (na.rational(0) <= na.rational(1)).if_else(Integer(1), d)
    assert isinstance(out_11, SecretInteger), type(out_11).__name__
    out_12 = (na.rational(0) <= na.rational(1)).if_else(UnsignedInteger(1), e)
    assert isinstance(out_12, SecretUnsignedInteger), type(out_12).__name__
    out_13 = (na.rational(0) <= na.rational(1)).if_else(
        UnsignedInteger(1), UnsignedInteger(0)
    )
    assert isinstance(out_13, PublicUnsignedInteger), type(out_13).__name__

    # Incompatible input types
    with pytest.raises(Exception):
        (a > Integer(1)).if_else(na.rational(0), na.rational(1))
    with pytest.raises(Exception):
        (Integer(1) > a).if_else(na.rational(0), na.rational(1))
    with pytest.raises(Exception):
        (a > d).if_else(na.rational(0), na.rational(1))
    with pytest.raises(Exception):
        (d > a).if_else(na.rational(0), na.rational(1))
    with pytest.raises(Exception):
        (na.rational(1) > Integer(1)).if_else(na.rational(0), na.rational(1))
    with pytest.raises(Exception):
        (Integer(1) > na.rational(1)).if_else(na.rational(0), na.rational(1))
    with pytest.raises(Exception):
        (na.rational(1) > d).if_else(na.rational(0), na.rational(1))
    with pytest.raises(Exception):
        (d > na.rational(1)).if_else(na.rational(0), na.rational(1))

    # Incompatible return types
    with pytest.raises(Exception):
        (a > b).if_else(c, Integer(1))
    with pytest.raises(Exception):
        (a > b).if_else(Integer(1), c)
    with pytest.raises(Exception):
        (a > b).if_else(c, d)
    with pytest.raises(Exception):
        (a > b).if_else(d, c)
    with pytest.raises(Exception):
        (na.rational(0) > na.rational(1)).if_else(c, Integer(1))
    with pytest.raises(Exception):
        (na.rational(0) > na.rational(1)).if_else(Integer(1), c)
    with pytest.raises(Exception):
        (na.rational(0) > na.rational(1)).if_else(c, d)
    with pytest.raises(Exception):
        (na.rational(0) > na.rational(1)).if_else(d, c)

    return (
        na.output(out_0, parties[2], "out_0")
        + na.output(out_1, parties[2], "out_1")
        + na.output(out_2, parties[2], "out_2")
        + na.output(out_3, parties[2], "out_3")
        + na.output(out_4, parties[2], "out_4")
        + na.output(out_5, parties[2], "out_5")
        + na.output(out_6, parties[2], "out_6")
        + na.output(out_7, parties[2], "out_7")
        + na.output(out_8, parties[2], "out_8")
        + na.output(out_9, parties[2], "out_9")
        + na.output(out_10, parties[2], "out_10")
        + na.output(out_11, parties[2], "out_11")
        + na.output(out_12, parties[2], "out_12")
        + na.output(out_13, parties[2], "out_13")
    )

import numpy as np
import pytest
from nada_dsl import *

import nada_algebra as na


def nada_main():
    parties = na.parties(3)

    a = na.array([3], parties[0], "A", na.SecretRational)
    b = na.array([3], parties[0], "B", SecretInteger)
    c = na.array([3], parties[0], "C", PublicInteger)

    # Mixed secretrational - secretinteger
    with pytest.raises(TypeError):
        na.NadaArray(np.array([a[0], b[0]]))
    # Mixed rational - secretinteger
    with pytest.raises(TypeError):
        na.NadaArray(np.array([na.rational(1), b[0]]))
    # Mixed secretrational - publicinteger
    with pytest.raises(TypeError):
        na.NadaArray(np.array([a[0], c[0]]))
    # Mixed rational - publicinteger
    with pytest.raises(TypeError):
        na.NadaArray(np.array([na.rational(1), c[0]]))
    # Mixed secretrational - publicinteger
    with pytest.raises(TypeError):
        na.NadaArray(np.array([a[0], c[0]]))

    # All-rational
    na.NadaArray(np.array([a[0], na.rational(1)]))
    # All-integer
    na.NadaArray(np.array([b[0], Integer(1)]))
    na.NadaArray(np.array([c[0], Integer(1)]))

    return a.output(parties[1], "my_output")

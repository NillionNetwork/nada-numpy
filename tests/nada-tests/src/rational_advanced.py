import pytest
import numpy as np
from nada_dsl import *
import nada_algebra as na


def nada_main():
    parties = na.parties(1)

    a = na.secret_rational("my_input_0", parties[0])
    b = na.rational(np.float16(1.2))

    out_0 = a.rescale_up().divide_no_rescale(b)
    out_1 = a.mul_no_rescale(b)
    out_2 = b.rescale_up().divide_no_rescale(a)
    out_3 = b.mul_no_rescale(a)

    na.set_log_scale(10)
    c = na.public_rational("my_input_1", parties[0])
    na.reset_log_scale()

    # Raise exception because different scaling
    with pytest.raises(Exception):
        a * c
    with pytest.raises(Exception):
        a.mul(c)
    with pytest.raises(Exception):
        a.mul_no_rescale(c)

    out_4 = a.mul(c, ignore_scale=True)
    out_5 = a.divide(c, ignore_scale=True)
    out_6 = a.mul_no_rescale(c, ignore_scale=True)
    out_7 = a.divide_no_rescale(c, ignore_scale=True)

    out_8 = b.mul(c, ignore_scale=True)
    out_9 = b.divide(c, ignore_scale=True)
    out_10 = b.mul_no_rescale(c, ignore_scale=True)
    out_11 = b.divide_no_rescale(c, ignore_scale=True)

    return [
        Output(out_0.value, "out_0", parties[0]),
        Output(out_1.value, "out_1", parties[0]),
        Output(out_2.value, "out_2", parties[0]),
        Output(out_3.value, "out_3", parties[0]),
        Output(out_4.value, "out_4", parties[0]),
        Output(out_5.value, "out_5", parties[0]),
        Output(out_6.value, "out_6", parties[0]),
        Output(out_7.value, "out_7", parties[0]),
        Output(out_8.value, "out_8", parties[0]),
        Output(out_9.value, "out_9", parties[0]),
        Output(out_10.value, "out_10", parties[0]),
        Output(out_11.value, "out_11", parties[0]),
    ]

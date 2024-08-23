import numpy as np
from nada_dsl import *

import nada_numpy as na


def nada_main():

    parties = na.parties(2)

    # We use na.SecretRational to create a secret rational number for party 0
    a = na.secret_rational("my_input_0", parties[0])

    c = na.NadaArray(np.array([a, na.rational(1.5)]))

    result_sign = c.sign()
    result_abs = c.abs()
    result_exp = c.exp()
    result_log = c.log()
    result_rec_NR = c.reciprocal(method="NR")
    result_rec_log = c.reciprocal(method="log")
    result_isqrt = c.inv_sqrt()
    result_sqrt = c.sqrt()
    result_sin = c.sin()
    result_cos = c.cos()
    result_tan = c.tan()
    result_tanh = c.tanh()
    result_tanh_che = c.tanh(method="chebyshev")
    result_tanh_motz = c.tanh(method="motzkin")
    result_sig = c.sigmoid()
    result_sig_che = c.sigmoid(method="chebyshev")
    result_sig_motz = c.sigmoid(method="motzkin")
    result_gelu = c.gelu()
    result_gelu_motz = c.gelu(method="motzkin")
    result_silu = c.silu()
    result_silu_che = c.silu(method_sigmoid="chebyshev")
    result_silu_motz = c.silu(method_sigmoid="motzkin")

    final_result = (
        result_sign.output(parties[1], "result_sign")
        + result_abs.output(parties[1], "result_abs")
        + result_exp.output(parties[1], "result_exp")
        + result_log.output(parties[1], "result_log")
        + result_rec_NR.output(parties[1], "result_rec_NR")
        + result_rec_log.output(parties[1], "result_rec_log")
        + result_isqrt.output(parties[1], "result_isqrt")
        + result_sqrt.output(parties[1], "result_sqrt")
        + result_sin.output(parties[1], "result_sin")
        + result_cos.output(parties[1], "result_cos")
        + result_tan.output(parties[1], "result_tan")
        + result_tanh.output(parties[1], "result_tanh")
        + result_tanh_che.output(parties[1], "result_tanh_che")
        + result_tanh_motz.output(parties[1], "result_tanh_motz")
        + result_sig.output(parties[1], "result_sig")
        + result_sig_che.output(parties[1], "result_sig_che")
        + result_sig_motz.output(parties[1], "result_sig_motz")
        + result_gelu.output(parties[1], "result_gelu")
        + result_gelu_motz.output(parties[1], "result_gelu_motz")
        + result_silu.output(parties[1], "result_silu")
        + result_silu_che.output(parties[1], "result_silu_che")
        + result_silu_motz.output(parties[1], "result_silu_motz")
    )

    return final_result

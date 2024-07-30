from nada_dsl import *

import nada_numpy as na


def nada_main():

    parties = na.parties(2)

    # We use na.SecretRational to create a secret rational number for party 0
    a = na.secret_rational("my_input_0", parties[0])

    result_exp = (a * na.rational(2)).exp()
    result_log = (a * na.rational(100)).log()
    result_rec_NR = (a * na.rational(2)).reciprocal(method="NR")
    result_rec_log = (a * na.rational(4)).reciprocal(method="log")
    result_isqrt = (a * na.rational(210)).inv_sqrt()
    result_sqrt = (a * na.rational(16)).sqrt()
    result_sin = (a * na.rational(2.1)).sin()
    result_cos = (a * na.rational(2.1)).cos()
    result_tan = (a * na.rational(4.8)).tan()
    result_tanh = (a * na.rational(1.3)).tanh()
    result_tanh_che = (a * na.rational(0.3)).tanh(method="chebyshev")
    result_tanh_motz = (a * na.rational(0.4)).tanh(method="motzkin")
    result_sig = (a * na.rational(0.1)).sigmoid()
    result_sig_che = (a * na.rational(-0.1)).sigmoid(method="chebyshev")
    result_sig_motz = (a * na.rational(10)).sigmoid(method="motzkin")
    result_gelu = (a * na.rational(-13)).gelu()
    result_gelu_motz = (a * na.rational(-13)).gelu(method="motzkin")
    result_silu = (a * na.rational(10)).silu()
    result_silu_che = (a * na.rational(-10)).silu(method_sigmoid="chebyshev")
    result_silu_motz = (a * na.rational(0)).silu(method_sigmoid="motzkin")

    return [
        Output(result_exp.value, "result_exp", parties[1]),
        Output(result_log.value, "result_log", parties[1]),
        Output(result_rec_NR.value, "result_rec_NR", parties[1]),
        Output(result_rec_log.value, "result_rec_log", parties[1]),
        Output(result_isqrt.value, "result_isqrt", parties[1]),
        Output(result_sqrt.value, "result_sqrt", parties[1]),
        Output(result_sin.value, "result_sin", parties[1]),
        Output(result_cos.value, "result_cos", parties[1]),
        Output(result_tan.value, "result_tan", parties[1]),
        Output(result_tanh.value, "result_tanh", parties[1]),
        Output(result_tanh_che.value, "result_tanh_che", parties[1]),
        Output(result_tanh_motz.value, "result_tanh_motz", parties[1]),
        Output(result_sig.value, "result_sig", parties[1]),
        Output(result_sig_che.value, "result_sig_che", parties[1]),
        Output(result_sig_motz.value, "result_sig_motz", parties[1]),
        Output(result_gelu.value, "result_gelu", parties[1]),
        Output(result_gelu_motz.value, "result_gelu_motz", parties[1]),
        Output(result_silu.value, "result_silu", parties[1]),
        Output(result_silu_che.value, "result_silu_che", parties[1]),
        Output(result_silu_motz.value, "result_silu_motz", parties[1]),
    ]

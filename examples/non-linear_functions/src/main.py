from nada_dsl import *
import nada_numpy as na
from fxpmath import (exp, log, reciprocal, inv_sqrt, sqrt,
                     cos, sin, tan, tanh, sigmoid, gelu, silu)

def nada_main():
    
    parties = na.parties(2)

    # We use na.SecretRational to create a secret rational number for party 0
    a = na.secret_rational("my_input_0", parties[0])

    result_exp = exp(a*na.rational(2))
    result_log = log(a*na.rational(100))
    result_rec_NR = reciprocal(a*na.rational(2), method="NR")
    result_rec_log = reciprocal(a*na.rational(4), method="log")
    result_isqrt = inv_sqrt(a*na.rational(210))
    result_sqrt = sqrt(a*na.rational(16))
    result_sin = sin(a*na.rational(2.1))
    result_cos = cos(a*na.rational(2.1))
    result_tan = tan(a*na.rational(4.8))
    result_tanh = tanh(a*na.rational(1.3))
    result_tanh_che = tanh(a*na.rational(0.3), method="chebyshev")
    result_tanh_motz = tanh(a*na.rational(0.4), method="motzkin")
    result_sig = sigmoid(a*na.rational(0.1))
    result_sig_che = sigmoid(a*na.rational(-0.1), method="chebyshev")
    result_sig_motz = sigmoid(a*na.rational(10), method="motzkin")
    result_gelu = gelu(a * na.rational(-13))
    result_gelu_motz = gelu(a * na.rational(-13), method="motzkin")
    result_silu = silu(a * na.rational(10))
    result_silu_che = silu(a * na.rational(-10), method_sigmoid="chebyshev")
    result_silu_motz = silu(a * na.rational(0), method_sigmoid="motzkin")


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
        Output(result_silu_motz.value, "result_silu_motz", parties[1])
    ]
from nada_dsl import *
import nada_algebra as na


def nada_main():
    parties = na.parties(1)

    a = na.Rational(Integer(3), scale=UnsignedInteger(16), is_scaled=False)

    b = na.Rational(Integer(2), scale=UnsignedInteger(16), is_scaled=False)
    c = SecretInteger(Input("my_input_0", parties[0]))
    d = Integer(1)
    e = na.SecretRational(c, scale=UnsignedInteger(16), is_scaled=False)

    out_0 = a < b
    out_1 = a <= b
    out_2 = a > b
    out_3 = a >= b
    out_4 = a == b

    out_5 = a < c
    out_6 = a <= c
    out_7 = a > c
    out_8 = a >= c
    out_9 = a == c

    out_10 = a < d
    out_11 = a <= d
    out_12 = a > d
    out_13 = a >= d
    out_14 = a == d

    out_15 = a < e
    out_16 = a <= e
    out_17 = a > e
    out_18 = a >= e
    out_19 = a == e


    return [
        Output(out_0, "my_output_0", parties[0]),
        Output(out_1, "my_output_1", parties[0]),
        Output(out_2, "my_output_2", parties[0]),
        Output(out_3, "my_output_3", parties[0]),
        Output(out_4, "my_output_4", parties[0]),
        Output(out_5, "my_output_5", parties[0]),
        Output(out_6, "my_output_6", parties[0]),
        Output(out_7, "my_output_7", parties[0]),
        Output(out_8, "my_output_8", parties[0]),
        Output(out_9, "my_output_9", parties[0]),
        Output(out_10, "my_output_10", parties[0]),
        Output(out_11, "my_output_11", parties[0]),
        Output(out_12, "my_output_12", parties[0]),
        Output(out_13, "my_output_13", parties[0]),
        Output(out_14, "my_output_14", parties[0]),
        Output(out_15, "my_output_15", parties[0]),
        Output(out_16, "my_output_16", parties[0]),
        Output(out_17, "my_output_17", parties[0]),
        Output(out_18, "my_output_18", parties[0]),
        Output(out_19, "my_output_19", parties[0]),
    ]

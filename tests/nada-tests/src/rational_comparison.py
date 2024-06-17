from nada_dsl import *

import nada_numpy as na


def nada_main():
    parties = na.parties(1)

    a = na.rational(3.2)
    b = na.rational(4.5)
    c = na.secret_rational("my_input_0", parties[0])  # 1.2
    d = na.public_rational("my_input_1", parties[0])  # 3.2

    out_0 = a < b  # True
    out_1 = a <= b  # True
    out_2 = a > b  # False
    out_3 = a >= b  # False
    out_4 = a == b  # False

    out_5 = a < c  # False
    out_6 = a <= c  # False
    out_7 = a > c  # True
    out_8 = a >= c  # True
    out_9 = a == c  # False

    out_10 = a == d  # True
    # out_11 = a != d # False
    out_12 = a <= d  # True
    out_13 = a >= d  # True
    out_14 = a > d  # False
    out_15 = a < d  # False

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
        # Output(out_11, "my_output_11", parties[0]),
        Output(out_12, "my_output_12", parties[0]),
        Output(out_13, "my_output_13", parties[0]),
        Output(out_14, "my_output_14", parties[0]),
        Output(out_15, "my_output_15", parties[0]),
    ]

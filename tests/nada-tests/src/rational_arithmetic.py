from nada_dsl import *

import nada_algebra as na


def nada_main():
    parties = na.parties(1)

    a = na.rational(3.2)
    b = na.public_rational("my_input_0", parties[0])  # 4.5
    c = na.secret_rational("my_input_1", parties[0])  # 1.2

    out_0 = a + b  # 7.7 -> 504627
    out_1 = a - b  # -1.3 -> -85197
    out_2 = a * b  # 14.4 -> 943717
    out_3 = a / b  # 0.7111111111111111 -> 46603

    out_4 = a + c  # 4.4 -> 288358
    out_5 = a - c  # 2.0 -> 131072
    out_6 = a * c  # 3.84 -> 251657
    out_7 = a / c  # 2.6666666666666665 -> 174763

    out_8 = a**8  # 10.24 -> 720568320
    out_9 = b**0  # 1 -> 65536
    out_10 = a ** (-3)  # 0.03051757812499999 -> 2000

    assert a == a**1, a**1

    return [
        Output(out_0.value, "my_output_0", parties[0]),
        Output(out_1.value, "my_output_1", parties[0]),
        Output(out_2.value, "my_output_2", parties[0]),
        Output(out_3.value, "my_output_3", parties[0]),
        Output(out_4.value, "my_output_4", parties[0]),
        Output(out_5.value, "my_output_5", parties[0]),
        Output(out_6.value, "my_output_6", parties[0]),
        Output(out_7.value, "my_output_7", parties[0]),
        Output(out_8.value, "my_output_8", parties[0]),
        Output(out_9.value, "my_output_9", parties[0]),
        Output(out_10.value, "my_output_10", parties[0]),
    ]

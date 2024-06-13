from nada_dsl import *

import nada_algebra as na


def nada_main():
    parties = na.parties(1)

    a = na.secret_rational("my_input_0", parties[0])  # 3.2 -> 209715
    b = na.secret_rational("my_input_1", parties[0])  # 4.5 -> 294912
    c = na.rational(1.2)  # 1.2 -> 78643

    out_0 = ((a + b - c) * b) / (a + b - c)  # b

    return [
        Output(out_0.value, "my_output_0", parties[0]),
    ]

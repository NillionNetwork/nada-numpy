from nada_dsl import *
import nada_algebra as na


def nada_main():
    parties = na.parties(2)

    a = na.array([3, 2], parties[0], "A", SecretInteger)
    b = na.array([3, 2], parties[0], "B", na.SecretRational)

    a_sum = a.sum()
    b_sum = b.sum()

    a_sum_arr = a.sum(axis=0)
    b_sum_arr = b.sum(axis=0)

    a_mean = a.mean()
    b_mean = b.mean()

    a_mean_arr = a.mean(axis=0)
    b_mean_arr = b.mean(axis=0)

    output_1 = (
        na.output(a_sum, parties[1], "a_sum")
        + na.output(a_mean, parties[1], "a_mean")
        + na.output(b_sum, parties[1], "b_sum")
        + na.output(b_mean, parties[1], "b_mean")
    )
    output_2 = (
        a_sum_arr.output(parties[1], "a_sum_arr")
        + b_sum_arr.output(parties[1], "b_sum_arr")
        + a_mean_arr.output(parties[1], "a_mean_arr")
        + b_mean_arr.output(parties[1], "b_mean_arr")
    )

    return output_1 + output_2

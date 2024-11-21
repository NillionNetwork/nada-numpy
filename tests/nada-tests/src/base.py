import numpy as np
from nada_dsl import *

import nada_numpy as na

DIM1 = 10
DIM2 = 4
DIM3 = 7

DIM4 = DIM1 + DIM2 + DIM3


def bubble_sort(array: na.NadaArray):
    size = DIM4
    for i in range(size):
        for j in range(0, size - i - 1):
            condition = array[j] > array[j + 1]
            res1 = condition.if_else(array[j + 1], array[j])
            res2 = condition.if_else(array[j], array[j + 1])
            array[j] = res1
            array[j + 1] = res2


def nada_main():
    party1 = Party(name="Party1")
    party1 = Party(name="Party2")
    party1 = Party(name="Party3")
    party4 = Party(name="Party4")

    array1 = na.array([DIM1], party1, "array1", SecretInteger)
    array2 = na.array([DIM2], party1, "array2", SecretInteger)
    array3 = na.array([DIM3], party1, "array3", SecretInteger)

    array4 = np.concatenate((np.concatenate((array1, array2)), array3))

    bubble_sort(array4)

    outs = []
    for i in range(DIM4):
        outs.append(Output(array4[i], "out_array_" + str(i), party4))

    return outs


if __name__ == "__main__":
    nada_main()

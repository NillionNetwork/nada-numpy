from nada_dsl import *
import nada_algebra as na


def nada_main():
    party = Party("party_0")

    a = SecretInteger(Input("a", party))

    ones1 = na.ones([2, 3])
    ones2 = na.ones_like(ones1)

    zeros1 = na.zeros([2, 3])
    zeros2 = na.zeros_like(zeros1)

    alphas1 = na.alphas([2, 3], alpha=a)
    alphas2 = na.alphas_like(alphas1, alpha=a)

    two_a = alphas1 + alphas2
    out = two_a + zeros1 + zeros2 + ones1 + ones2

    return out.output(party, "my_output")

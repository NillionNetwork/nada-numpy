# Rational Numbers Tutorial

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NillionNetwork/nada-numpy/blob/main/examples/rational_numbers/rational_numbers.ipynb)

This tutorial shows how to use Nada Numpy Rational datatypes to work with fixed-point numbers in Nada.

## 💡 Notions

This tutorial uses fixed point numbers as it is the only available way to use Fixed-Point numbers in Nada. The representation of a fixed point number uses integer places to represent decimal digits. Thus, every number is multiplied by a scaling factor, that we refer to as `SCALE` ($\Delta = 2^{16}$) or `LOG_SCALE` in its logarithmic notation ($log_2\Delta = 16$). In a nutshell, this means we will use 16 bits to represent decimals. 

If we want to input a variable `a = float(3.2)`, we need to first encode it. For that we will define a new variable `a'` which is going to be the scaled version. In this case, the scaling factor (to simplify) is going to by 3 bits so, $log_2\Delta = 3$ and $\Delta = 2^3 = 8$. With the following formula, we compute the encoded value:

$$ a' = round(a * \Delta) = round(a * 2^{log_2\Delta}) = 3.2 \cdot 2^3 = 3.2 \cdot 8 = 26 $$

Thus, in order to introduce a value with 3 bits of precision, we would be inputing 26 instead of 3.2.

## 🚨 Limitations
The choice for blind computing implies certain trade-offs in comparison to conventional computing. What you gain in privacy, you pay in extra computational overhead & capacity constraints.

Therefore, you will notice that large-scale computational workloads may lead to long compilation and/or execution times or hitting network capacity guardrails.

That said, the Nillion team is working around the clock to push the boundaries of this technology and bring the potential of blind computing to reality 🚀

## ➡️ Stay in touch
If you want to get involved in the blind computing community and be the first to know all big updates, join our Discord

[![Discord](https://img.shields.io/badge/Discord-nillionnetwork-%235865F2?logo=discord)](https://discord.gg/nillionnetwork)

And if you want to contribute to the blind computing revolution, we welcome open-source contributors!

[![GitHub Discussions](https://img.shields.io/badge/GitHub_Discussions-NillionNetwork-%23181717?logo=github)](https://github.com/orgs/NillionNetwork/discussions)


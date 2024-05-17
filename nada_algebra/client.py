import py_nillion_client as nillion
import numpy as np

def parties(num: int, prefix: str = "Party"):
    """
    Create a list of Party default name objects.

    Args:
        num (`int`): The number of parties to create.
        prefix (`str`, optional): The prefix to use for party names. Defaults to "Party".

    Returns:
        `list`: A list of Party objects with names in the format "{prefix}{i}".
    """
    return [f"{prefix}{i}" for i in range(num)]


def array(arr: np.ndarray, prefix: str, nada_type: nillion.SecretInteger | nillion.SecretUnsignedInteger | nillion.PublicVariableInteger | nillion.PublicVariableUnsignedInteger = nillion.SecretInteger) -> dict:
    """
    Recursively generates a list of nillion input objects for each element in the Secret input array.

    Args:
        arr (`np.ndarray`): The input array.
        prefix (`str`): The prefix to be added to the Output names.
        nada_type (`Union[nillion.SecretInteger, nillion.SecretUnsignedInteger]`): The type of the values introduced. Defaults to SecretInteger.

    Returns:
        `dict`: The output dictionary
    """
    if len(arr.shape) == 1:
        return {f"{prefix}_{i}": nada_type(int(arr[i])) for i in range(arr.shape[0])}
    return {k:v for i in range(arr.shape[0]) for (k, v) in array(arr[i], f"{prefix}_{i}", nada_type).items()}

def concat(list_dict: list):
    """
    Combines a list of non-overlapping dictionaries into a single dictionary.

    WARN: It will overwrite the values of the keys if there are common keys.

    Args:
        list_dict (list): A list of dictionaries.

    Returns:
        dict: A single dictionary.
    """
    return {k: v for d in list_dict for (k, v) in d.items()}
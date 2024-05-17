from nada_dsl import Party, SecretInteger, SecretUnsignedInteger, PublicInteger, PublicUnsignedInteger, Integer, UnsignedInteger
from nada_algebra.array import NadaArray
import numpy as np


def parties(num: int, prefix: str = "Party"):
    """
    Create a list of Party objects.

    Args:
        num (int): The number of parties to create.
        prefix (str, optional): The prefix to use for party names. Defaults to "Party".

    Returns:
        list: A list of Party objects with names in the format "{prefix}{i}".
    """
    return [Party(name=f"{prefix}{i}") for i in range(num)]

def __from_list(lst: list, nada_type: Integer | UnsignedInteger):
    if len(lst.shape) == 1:
        return [nada_type(int(elem)) for elem in lst]
    return [__from_list(lst[i], nada_type) for i in range(len(lst))]

def from_list(lst: list, nada_type: Integer | UnsignedInteger = Integer) -> NadaArray:
    """
    Create a cleartext NadaArray with the specified elements.

    Args:
        lst (list): A list of integers representing the elements of the array.

    Returns:
        np.ndarray: The created NumPy array.
    """
    if not isinstance(lst, np.ndarray):
        lst = np.array(lst)
    return NadaArray(np.array(__from_list(lst, nada_type)))

def ones(dims: list, nada_type: Integer | UnsignedInteger = Integer) -> NadaArray:
    """
    Create a cleartext NadaArray with ones of dims dimensions.

    Args:
        dims (list): A list of integers representing the dimensions of the array.

    Returns:
        np.ndarray: The created NumPy array.
    """
    return from_list(np.ones(dims))

def zeros(dims: list, nada_type: Integer | UnsignedInteger = Integer) -> NadaArray:
    """
    Create a cleartext NadaArray with zeros of dims dimensions.

    Args:
        dims (list): A list of integers representing the dimensions of the array.

    Returns:
        np.ndarray: The created NumPy array.
    """ 
    return from_list(np.zeros(dims))

def array(dims: list, party: Party, prefix: str, nada_type: SecretInteger | SecretUnsignedInteger | PublicInteger | PublicUnsignedInteger = SecretInteger) -> "NadaArray":
    """
    Create a NumPy array with the specified dimensions for specified nada_type.

    Args:
        dims (list): A list of integers representing the dimensions of the array.
        party (Party): An object representing the party.
        prefix (str): A string prefix for the array elements.
        nada_type (type): 

    Returns:
        np.ndarray: The created NumPy array.
    """    
    return NadaArray.array(dims, party, prefix, nada_type)

def random(dims: list, nada_type: SecretInteger | SecretUnsignedInteger = SecretInteger) -> "NadaArray":
    """
    Create a random array with the specified dimensions.

    Args:
        dims (list): A list of integers representing the dimensions of the array.
        party (Party): The party object representing the current party.
        prefix (str): A prefix string to be used for generating random values.

    Returns:
        np.ndarray: A NumPy array with random SecretInteger values.

    """
    return NadaArray.random(dims, nada_type)

def output(array: NadaArray, party: Party, prefix: str):
    """
    Recursively generates a list of Output objects for each element in the input NadaArray.

    Args:
        array (np.ndarray): The input Nada array.
        party (Party): The party object.
        prefix (str): The prefix to be added to the Output names.

    Returns:
        List[Output]: A list of Output objects.
    """    
    return array.output(party, prefix)

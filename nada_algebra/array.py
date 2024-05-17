from typing import Callable, Union
import numpy as np
from dataclasses import dataclass
from nada_dsl import Input, Party, Output, SecretInteger, SecretUnsignedInteger, PublicInteger, PublicUnsignedInteger, Integer, UnsignedInteger

@dataclass
class NadaArray:
    """
    Represents an array-like object with additional functionality.

    Attributes:
        inner (np.ndarray): The underlying NumPy array.
    """

    inner: np.ndarray

    def __add__(self, other: Union["NadaArray", np.ndarray, int, Integer, UnsignedInteger, SecretInteger, SecretUnsignedInteger, PublicInteger, PublicUnsignedInteger]) -> "NadaArray":
        """
        Element-wise addition with broadcasting.

        Args:
            other (Union[NadaArray, np.ndarray, int, Integer, UnsignedInteger, SecretInteger, SecretUnsignedInteger, PublicInteger, PublicUnsignedInteger]): The object to add.
        Returns:
            NadaArray: A new NadaArray representing the element-wise addition result.
        """
        if isinstance(other, NadaArray):
            return NadaArray(self.inner + other.inner)
        elif isinstance(other, int):
            return NadaArray(self.inner + Integer(other))
        return NadaArray(self.inner + other)

    def __sub__(self, other: Union["NadaArray", np.ndarray, int, Integer, UnsignedInteger, SecretInteger, SecretUnsignedInteger, PublicInteger, PublicUnsignedInteger]) -> "NadaArray":
        """
        Element-wise subtraction with broadcasting.

        Args:
            other (Union[NadaArray, np.ndarray, int, Integer, UnsignedInteger, SecretInteger, SecretUnsignedInteger, PublicInteger, PublicUnsignedInteger]): The object to subtract.

        Returns:
            NadaArray: A new NadaArray representing the element-wise subtraction result.
        """
        if isinstance(other, NadaArray):
            return NadaArray(self.inner - other.inner)
        elif isinstance(other, int):
            return NadaArray(self.inner - Integer(other))
        return NadaArray(self.inner - other)

    def __mul__(self, other: Union["NadaArray", np.ndarray, int, Integer, UnsignedInteger, SecretInteger, SecretUnsignedInteger, PublicInteger, PublicUnsignedInteger]) -> "NadaArray":
        """
        Element-wise multiplication with broadcasting.

        Args:
            other (Union[NadaArray, np.ndarray, int, Integer, UnsignedInteger, SecretInteger, SecretUnsignedInteger, PublicInteger, PublicUnsignedInteger]): The object to multiply.

        Returns:
            NadaArray: A new NadaArray representing the element-wise multiplication result.
        """
        if isinstance(other, NadaArray):
            return NadaArray(self.inner * other.inner)
        elif isinstance(other, int):
            return NadaArray(self.inner * Integer(other))
        return NadaArray(self.inner * other)

    def __truediv__(self, other: Union["NadaArray", np.ndarray, int, Integer, UnsignedInteger, SecretInteger, SecretUnsignedInteger, PublicInteger, PublicUnsignedInteger]) -> "NadaArray":
        """
        Element-wise division with broadcasting.

        Args:
            other (Union[NadaArray, np.ndarray, int, Integer, UnsignedInteger, SecretInteger, SecretUnsignedInteger, PublicInteger, PublicUnsignedInteger]): The object to divide.

        Returns:
            NadaArray: A new NadaArray representing the element-wise division result.
        """
        if isinstance(other, NadaArray):
            return NadaArray(self.inner / other.inner)
        elif isinstance(other, int):
            return NadaArray(self.inner / Integer(other))
        return NadaArray(self.inner / other)
    
    def dot(self, other: "NadaArray") -> "NadaArray":
        """
        Computes the dot product between two NadaArray objects.

        Args:
            other (NadaArray): The NadaArray to compute dot product with.

        Returns:
            NadaArray: A new NadaArray representing the dot product result.
        """
        return NadaArray(self.inner.dot(other.inner))

    def sum(self) -> SecretInteger | SecretUnsignedInteger:
        """
        Computes the variance of the elements in the array.

        Returns:
            NadaType: The mean of the array in the type corresponding to the array
        """
        return NadaArray(self.inner.sum())

    def hstack(self, other: "NadaArray") -> "NadaArray":
        """
        Horizontally stacks two NadaArray objects.

        Args:
            other (NadaArray): The NadaArray to stack horizontally with.

        Returns:
            NadaArray: A new NadaArray representing the horizontal stack.
        """
        return NadaArray(np.hstack((self.inner, other.inner)))

    def vstack(self, other: "NadaArray") -> "NadaArray":
        """
        Vertically stacks two NadaArray objects.

        Args:
            other (NadaArray): The NadaArray to stack vertically with.

        Returns:
            NadaArray: A new NadaArray representing the vertical stack.
        """
        return NadaArray(np.vstack((self.inner, other.inner)))

    def reveal(self) -> "NadaArray":
        """
        Recursively reveals the elements of the array.

        Returns:
            NadaArray: A new NadaArray with revealed values.
        """
        return self.applypyfunc(lambda x: x.reveal())

    @staticmethod
    def apply_function_elementwise(func: Callable, array: np.ndarray) -> np.ndarray:
        """
        Applies a function element-wise to the input array.

        Args:
            func (Callable): The function to apply.
            array (np.ndarray): The input array.

        Returns:
            np.ndarray: A NumPy array with the applied function.
        """
        if len(array.shape) == 1:
            return [func(x) for x in array]
        return [NadaArray.apply_function_elementwise(func, array[i]) for i in range(array.shape[0])]
    
    def applypyfunc(self, func: Callable) -> "NadaArray":
        """
        Applies a Python function element-wise to the array.

        Args:
            func (Callable): The function to apply.

        Returns:
            NadaArray: A new NadaArray with the applied function.
        """
        return NadaArray(np.array(NadaArray.apply_function_elementwise(func, self.inner)))
    
    def output_array(array: np.ndarray, party: Party, prefix: str):
        """
        Recursively generates a list of Output objects for each element in the input array.

        Args:
            array (np.ndarray): The input array.
            party (Party): The party object.
            prefix (str): The prefix to be added to the Output names.

        Returns:
            List[Output]: A list of Output objects.
        """
        if isinstance(array, SecretInteger) or isinstance(array, SecretUnsignedInteger):
            return [Output(array, f"{prefix}_0", party)]
        
        if len(array.shape) == 1:
            return [Output(array[i], f"{prefix}_{i}", party) for i in range(array.shape[0])]
        return [v for i in range(array.shape[0]) for v in NadaArray.output_array(array[i], party, f"{prefix}_{i}")]
    
    def output(self, party: Party, prefix: str):
        """
        Recursively generates a list of Output objects for each element in the input NadaArray.

        Args:
            array (np.ndarray): The input Nada array.
            party (Party): The party object.
            prefix (str): The prefix to be added to the Output names.

        Returns:
            List[Output]: A list of Output objects.
        """
        return NadaArray.output_array(self.inner, party, prefix)
    
    def create_list(dims: list, party: Party, prefix: str, generator: Callable) -> list:
        """
        Recursively creates a nested list of objects generated by generator function

        Args:
            dims (list): A list of integers representing the dimensions of the desired nested list.
            party (Party): The party object representing the party to which the SecretInteger objects belong.
            prefix (str): A string prefix to be used in the names of the SecretInteger objects.

        Returns:
            list: A nested list of SecretInteger objects.

        """
        if len(dims) == 1:
            return [generator(f"{prefix}_{i}", party) for i in range(dims[0])]
        return [NadaArray.create_list(dims[1:], party, f"{prefix}_{i}", generator) for i in range(dims[0])]

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
        secret_int_generator = lambda name, party: nada_type(Input(name=name, party=party))
        return NadaArray(np.array(NadaArray.create_list(dims, party, prefix, secret_int_generator)))
    

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
        secret_random_generator = lambda name, party: nada_type.random() 
        return NadaArray(np.array(NadaArray.create_list(dims, None, None, secret_random_generator)))


    
"""
    This module provides an enhanced array class, NadaArray, with additional functionality
    for mathematical operations and integration with the Nada Algebra library.
"""

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
from nada_dsl import (
    Input,
    Party,
    Output,
    SecretInteger,
    SecretUnsignedInteger,
    PublicInteger,
    PublicUnsignedInteger,
    Integer,
    UnsignedInteger,
)


@dataclass
class NadaArray:
    """
    Represents an array-like object with additional functionality.

    Attributes:
        inner (np.ndarray): The underlying NumPy array.
    """

    inner: np.ndarray

    def __add__(
        self,
        other: Union[
            "NadaArray",
            np.ndarray,
            int,
            Integer,
            UnsignedInteger,
            SecretInteger,
            SecretUnsignedInteger,
            PublicInteger,
            PublicUnsignedInteger,
        ],
    ) -> "NadaArray":
        """
        Perform element-wise addition with broadcasting.

        Args:
            other (Union[NadaArray, np.ndarray, int, Integer, UnsignedInteger, SecretInteger,
                SecretUnsignedInteger, PublicInteger, PublicUnsignedInteger]): The object to add.

        Returns:
            NadaArray: A new NadaArray representing the element-wise addition result.
        """
        if isinstance(other, NadaArray):
            return NadaArray(self.inner + other.inner)
        if isinstance(other, int):
            return NadaArray(self.inner + Integer(other))
        return NadaArray(self.inner + other)

    def __sub__(
        self,
        other: Union[
            "NadaArray",
            np.ndarray,
            int,
            Integer,
            UnsignedInteger,
            SecretInteger,
            SecretUnsignedInteger,
            PublicInteger,
            PublicUnsignedInteger,
        ],
    ) -> "NadaArray":
        """
        Perform element-wise subtraction with broadcasting.

        Args:
            other (Union[NadaArray, np.ndarray, int,
                Integer, UnsignedInteger, SecretInteger,
                SecretUnsignedInteger, PublicInteger,
                PublicUnsignedInteger]): The object to subtract.

        Returns:
            NadaArray: A new NadaArray representing the element-wise subtraction result.
        """
        if isinstance(other, NadaArray):
            return NadaArray(self.inner - other.inner)
        if isinstance(other, int):
            return NadaArray(self.inner - Integer(other))
        return NadaArray(self.inner - other)

    def __mul__(
        self,
        other: Union[
            "NadaArray",
            np.ndarray,
            int,
            Integer,
            UnsignedInteger,
            SecretInteger,
            SecretUnsignedInteger,
            PublicInteger,
            PublicUnsignedInteger,
        ],
    ) -> "NadaArray":
        """
        Perform element-wise multiplication with broadcasting.

        Args:
            other (Union[NadaArray, np.ndarray, int,
                Integer, UnsignedInteger, SecretInteger,
                SecretUnsignedInteger, PublicInteger,
                PublicUnsignedInteger]): The object to multiply.

        Returns:
            NadaArray: A new NadaArray representing the element-wise multiplication result.
        """
        if isinstance(other, NadaArray):
            return NadaArray(self.inner * other.inner)
        if isinstance(other, int):
            return NadaArray(self.inner * Integer(other))
        return NadaArray(self.inner * other)

    def __truediv__(
        self,
        other: Union[
            "NadaArray",
            np.ndarray,
            int,
            Integer,
            UnsignedInteger,
            SecretInteger,
            SecretUnsignedInteger,
            PublicInteger,
            PublicUnsignedInteger,
        ],
    ) -> "NadaArray":
        """
        Perform element-wise division with broadcasting.

        Args:
            other (Union[NadaArray, np.ndarray, int, Integer, UnsignedInteger, SecretInteger,
                SecretUnsignedInteger, PublicInteger, PublicUnsignedInteger]): The object to divide.

        Returns:
            NadaArray: A new NadaArray representing the element-wise division result.
        """
        if isinstance(other, NadaArray):
            return NadaArray(self.inner / other.inner)
        if isinstance(other, int):
            return NadaArray(self.inner / Integer(other))
        return NadaArray(self.inner / other)

    def __matmul__(self, other: "NadaArray") -> "NadaArray":
        """
        Perform matrix multiplication with another NadaArray.

        Args:
            other (NadaArray): The NadaArray to perform matrix multiplication with.

        Returns:
            NadaArray: A new NadaArray representing the result of matrix multiplication.
        """
        if isinstance(other, NadaArray):
            return NadaArray(self.inner @ other.inner)

    def dot(self, other: "NadaArray") -> "NadaArray":
        """
        Compute the dot product between two NadaArray objects.

        Args:
            other (NadaArray): The NadaArray to compute dot product with.

        Returns:
            NadaArray: A new NadaArray representing the dot product result.
        """
        return NadaArray(self.inner.dot(other.inner))

    def sum(self) -> Union[SecretInteger, SecretUnsignedInteger]:
        """
        Compute the sum of the elements in the array.

        Returns:
            Union[SecretInteger, SecretUnsignedInteger]: The sum of the array elements.
        """
        return NadaArray(self.inner.sum())

    def hstack(self, other: "NadaArray") -> "NadaArray":
        """
        Horizontally stack two NadaArray objects.

        Args:
            other (NadaArray): The NadaArray to stack horizontally with.

        Returns:
            NadaArray: A new NadaArray representing the horizontal stack.
        """
        return NadaArray(np.hstack((self.inner, other.inner)))

    def vstack(self, other: "NadaArray") -> "NadaArray":
        """
        Vertically stack two NadaArray objects.

        Args:
            other (NadaArray): The NadaArray to stack vertically with.

        Returns:
            NadaArray: A new NadaArray representing the vertical stack.
        """
        return NadaArray(np.vstack((self.inner, other.inner)))

    def reveal(self) -> "NadaArray":
        """
        Reveal the elements of the array.

        Returns:
            NadaArray: A new NadaArray with revealed values.
        """
        return self.applypyfunc(lambda x: x.reveal())

    @staticmethod
    def apply_function_elementwise(func: Callable, array: np.ndarray) -> np.ndarray:
        """
        Apply a function element-wise to the input array.

        Args:
            func (Callable): The function to apply.
            array (np.ndarray): The input array.

        Returns:
            np.ndarray: A NumPy array with the function applied to each element.
        """
        if len(array.shape) == 1:
            return [func(x) for x in array]
        return [
            NadaArray.apply_function_elementwise(func, array[i])
            for i in range(array.shape[0])
        ]

    def applypyfunc(self, func: Callable) -> "NadaArray":
        """
        Apply a Python function element-wise to the array.

        Args:
            func (Callable): The function to apply.

        Returns:
            NadaArray: A new NadaArray with the function applied to each element.
        """
        return NadaArray(
            np.array(NadaArray.apply_function_elementwise(func, self.inner))
        )

    @staticmethod
    def output_array(array: np.ndarray, party: Party, prefix: str) -> list:
        """
        Generate a list of Output objects for each element in the input array.

        Args:
            array (np.ndarray): The input array.
            party (Party): The party object.
            prefix (str): The prefix to be added to the Output names.

        Returns:
            list: A list of Output objects.
        """
        if isinstance(array, (SecretInteger, SecretUnsignedInteger)):
            return [Output(array, f"{prefix}_0", party)]

        if len(array.shape) == 1:
            return [
                Output(array[i], f"{prefix}_{i}", party) for i in range(array.shape[0])
            ]
        return [
            v
            for i in range(array.shape[0])
            for v in NadaArray.output_array(array[i], party, f"{prefix}_{i}")
        ]

    def output(self, party: Party, prefix: str) -> list:
        """
        Generate a list of Output objects for each element in the NadaArray.

        Args:
            party (Party): The party object.
            prefix (str): The prefix for naming the Output objects.

        Returns:
            list: A list of Output objects.
        """
        return NadaArray.output_array(self.inner, party, prefix)

    @staticmethod
    def create_list(dims: list, party: Party, prefix: str, generator: Callable) -> list:
        """
        Recursively create a nested list of objects generated by a generator function.

        Args:
            dims (list): A list of integers representing the dimensions of the desired nested list.
            party (Party): The party object representing the party to which the elements belong.
            prefix (str): A prefix for the names of the elements.
            generator (Callable): The function to generate the elements.

        Returns:
            list: A nested list of generated objects.
        """
        if len(dims) == 1:
            return [generator(f"{prefix}_{i}", party) for i in range(dims[0])]
        return [
            NadaArray.create_list(
                dims[1:],
                party,
                f"{prefix}_{i}",
                generator,
            )
            for i in range(dims[0])
        ]

    @staticmethod
    def array(
        dims: list,
        party: Party,
        prefix: str,
        nada_type: Union[
            SecretInteger, SecretUnsignedInteger, PublicInteger, PublicUnsignedInteger
        ] = SecretInteger,
    ) -> "NadaArray":
        """
        Create a NadaArray with the specified dimensions and element type.

        Args:
            dims (list): A list of integers representing the dimensions of the array.
            party (Party): The party object.
            prefix (str): A prefix for naming the array elements.
            nada_type (type, optional): The type of elements to create. Defaults to SecretInteger.

        Returns:
            NadaArray: The created NadaArray.
        """
        return NadaArray(
            np.array(
                NadaArray.create_list(
                    dims,
                    party,
                    prefix,
                    lambda name, party: nada_type(Input(name=name, party=party)),
                )
            )
        )

    @staticmethod
    def random(
        dims: list,
        nada_type: Union[SecretInteger, SecretUnsignedInteger] = SecretInteger,
    ) -> "NadaArray":
        """
        Create a random NadaArray with the specified dimensions.

        Args:
            dims (list): A list of integers representing the dimensions of the array.
            nada_type (type, optional): The type of elements to create. Defaults to SecretInteger.

        Returns:
            NadaArray: The created random NadaArray.
        """
        return NadaArray(
            np.array(
                NadaArray.create_list(
                    dims, None, None, lambda name, party: nada_type.random()
                )
            )
        )

"""
    This module provides an enhanced array class, NadaArray, with additional functionality
    for mathematical operations and integration with the Nada Algebra library.
"""

from dataclasses import dataclass
from types import NoneType
from typing import Any, Callable, Type, Union

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
from nada_algebra.types import (
    Rational,
    SecretRational,
    public_rational,
    secret_rational,
    get_log_scale,
)


@dataclass
class NadaArray:
    """
    Represents an array-like object with additional functionality.

    Attributes:
        inner (np.ndarray): The underlying NumPy array.
    """

    inner: np.ndarray

    SUPPORTED_OPERATIONS = {
        "base",
        "compress",
        "copy",
        "cumprod",
        "cumsum",
        "data",
        "diagonal",
        "fill",
        "flags",
        "flat",
        "flatten",
        "item",
        "itemset",
        "itemsize",
        "nbytes",
        "ndim",
        "prod",
        "put",
        "ravel",
        "repeat",
        "reshape",
        "resize",
        "shape",
        "size",
        "squeeze",
        "strides",
        "sum",
        "swapaxes",
        "T",
        "take",
        "tolist",
        "trace",
        "transpose",
    }

    def __getitem__(self, item):
        """
        Get an item from the array.

        Args:
            item: The item to retrieve.

        Returns:
            NadaArray: A new NadaArray representing the retrieved item.
        """
        if len(self.inner.shape) == 1:
            return self.inner[item]
        return NadaArray(self.inner[item])

    def __setitem__(self, key, value):
        """
        Set an item in the array.

        Args:
            key: The key to set.
            value: The value to set.
        """
        if isinstance(value, NadaArray):
            self.inner[key] = value.inner
        else:
            self.inner[key] = value

    def add(self, other: Any) -> "NadaArray":
        """
        Perform element-wise addition with broadcasting.

        Args:
            other (Any): The object to add.

        Returns:
            NadaArray: A new NadaArray representing the element-wise addition result.
        """
        if isinstance(other, NadaArray):
            return NadaArray(self.inner + other.inner)
        return NadaArray(self.inner + other)

    def __add__(self, other: Any) -> "NadaArray":
        """
        Perform element-wise addition with broadcasting.

        Args:
            other (Any): The object to add.

        Returns:
            NadaArray: A new NadaArray representing the element-wise addition result.
        """
        return self.add(other)

    def __iadd__(self, other: Any) -> "NadaArray":
        """
        Perform element-wise inplace addition with broadcasting.

        Args:
            other (Any): The object to add.

        Returns:
            NadaArray: A new NadaArray representing the element-wise addition result.
        """
        return self.add(other)

    def sub(self, other: Any) -> "NadaArray":
        """
        Perform element-wise subtraction with broadcasting.

        Args:
            other (Any): The object to subtract.

        Returns:
            NadaArray: A new NadaArray representing the element-wise subtraction result.
        """
        if isinstance(other, NadaArray):
            return NadaArray(self.inner - other.inner)
        return NadaArray(self.inner - other)

    def __sub__(self, other: Any) -> "NadaArray":
        """
        Perform element-wise subtraction with broadcasting.

        Args:
            other (Any): The object to subtract.

        Returns:
            NadaArray: A new NadaArray representing the element-wise subtraction result.
        """
        return self.sub(other)

    def __isub__(self, other: Any) -> "NadaArray":
        """
        Perform element-wise inplace subtraction with broadcasting.

        Args:
            other (Any): The object to subtract.

        Returns:
            NadaArray: A new NadaArray representing the element-wise subtraction result.
        """
        return self.sub(other)

    def mul(self, other: Any) -> "NadaArray":
        """
        Perform element-wise multiplication with broadcasting.

        Args:
            other (Any): The object to multiply.

        Returns:
            NadaArray: A new NadaArray representing the element-wise multiplication result.
        """
        if isinstance(other, NadaArray):
            return NadaArray(self.inner * other.inner)
        return NadaArray(self.inner * other)

    def __mul__(self, other: Any) -> "NadaArray":
        """
        Perform element-wise multiplication with broadcasting.

        Args:
            other (Any): The object to multiply.

        Returns:
            NadaArray: A new NadaArray representing the element-wise multiplication result.
        """
        return self.mul(other)

    def __imul__(self, other: Any) -> "NadaArray":
        """
        Perform element-wise inplace multiplication with broadcasting.

        Args:
            other (Any): The object to multiply.

        Returns:
            NadaArray: A new NadaArray representing the element-wise multiplication result.
        """
        return self.mul(other)

    def __pow__(self, other: int) -> "NadaArray":
        """
        Raises NadaArray to a power.

        Args:
            other (int): Power value.

        Returns:
            NadaArray: Result NadaArray.
        """
        return self.apply(lambda x: x**other)

    def divide(self, other: Any) -> "NadaArray":
        """
        Perform element-wise division with broadcasting.

        Args:
            other (Any): The object to divide.

        Returns:
            NadaArray: A new NadaArray representing the element-wise division result.
        """
        if isinstance(other, NadaArray):
            return NadaArray(self.inner / other.inner)
        return NadaArray(self.inner / other)

    def __truediv__(self, other: Any) -> "NadaArray":
        """
        Perform element-wise division with broadcasting.

        Args:
            other (Any): The object to divide.

        Returns:
            NadaArray: A new NadaArray representing the element-wise division result.
        """
        return self.divide(other)

    def __itruediv__(self, other: Any) -> "NadaArray":
        """
        Perform element-wise inplace division with broadcasting.

        Args:
            other (Any): The object to divide.

        Returns:
            NadaArray: A new NadaArray representing the element-wise division result.
        """
        return self.divide(other)

    def matmul(self, other: "NadaArray") -> "NadaArray":
        """
        Perform matrix multiplication with another NadaArray.

        Args:
            other (NadaArray): The NadaArray to perform matrix multiplication with.

        Returns:
            NadaArray: A new NadaArray representing the result of matrix multiplication.
        """
        if isinstance(other, NadaArray):
            return NadaArray(np.array(self.inner @ other.inner))
        return NadaArray(np.array(self.inner @ other))

    def __matmul__(self, other: Any) -> "NadaArray":
        """
        Perform matrix multiplication with another NadaArray.

        Args:
            other (NadaArray): The NadaArray to perform matrix multiplication with.

        Returns:
            NadaArray: A new NadaArray representing the result of matrix multiplication.
        """
        return self.matmul(other)

    def __imatmul__(self, other: Any) -> "NadaArray":
        """
        Perform inplace matrix multiplication with another NadaArray.

        Args:
            other (NadaArray): The NadaArray to perform matrix multiplication with.

        Returns:
            NadaArray: A new NadaArray representing the result of matrix multiplication.
        """
        return self.matmul(other)

    def dot(self, other: "NadaArray") -> "NadaArray":
        """
        Compute the dot product between two NadaArray objects.

        Args:
            other (NadaArray): The NadaArray to compute dot product with.

        Returns:
            NadaArray: A new NadaArray representing the dot product result.
        """
        return NadaArray(self.inner.dot(other.inner))

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
        return self.apply(lambda x: x.reveal())

    def apply(self, func: Callable[[Any], Any]) -> "NadaArray":
        """
        Apply a Python function element-wise to the array.

        Args:
            func (Callable[[Any], Any]): The function to apply.

        Returns:
            NadaArray: A new NadaArray with the function applied to each element.
        """
        return NadaArray(np.frompyfunc(func, 1, 1)(self.inner))

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
        if isinstance(
            array,
            (
                SecretInteger,
                SecretUnsignedInteger,
                PublicInteger,
                PublicUnsignedInteger,
                Integer,
                UnsignedInteger,
            ),
        ):
            return [Output(array, f"{prefix}_0", party)]
        elif isinstance(array, (Rational, SecretRational)):
            return [Output(array.value, f"{prefix}_0", party)]

        if len(array.shape) == 1:
            return [
                (
                    Output(array[i].value, f"{prefix}_{i}", party)
                    if isinstance(array[i], (Rational, SecretRational))
                    else Output(array[i], f"{prefix}_{i}", party)
                )
                for i in range(array.shape[0])
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
            SecretInteger,
            SecretUnsignedInteger,
            PublicInteger,
            PublicUnsignedInteger,
            SecretRational,
            Rational,
        ],
    ) -> "NadaArray":
        """
        Create a NadaArray with the specified dimensions and element type.

        Args:
            dims (list): A list of integers representing the dimensions of the array.
            party (Party): The party object.
            prefix (str): A prefix for naming the array elements.
            nada_type (type): The type of elements to create.

        Returns:
            NadaArray: The created NadaArray.

        Raises:
            ValueError: Raised if the nada_type is not supported.
        """
        generator = None
        if nada_type == Rational:
            generator = lambda name, party: public_rational(name=name, party=party)
        elif nada_type == SecretRational:
            generator = lambda name, party: secret_rational(name=name, party=party)
        elif nada_type in (
            SecretInteger,
            SecretUnsignedInteger,
            PublicInteger,
            PublicUnsignedInteger,
        ):
            generator = lambda name, party: nada_type(Input(name=name, party=party))
        else:
            raise ValueError(f"Unsupported nada_type: {nada_type}")

        return NadaArray(
            np.array(NadaArray.create_list(dims, party, prefix, generator))
        )

    @staticmethod
    def random(
        dims: list,
        nada_type: Union[
            SecretInteger, SecretUnsignedInteger, SecretRational
        ] = SecretInteger,
    ) -> "NadaArray":
        """
        Create a random NadaArray with the specified dimensions.

        Args:
            dims (list): A list of integers representing the dimensions of the array.
            nada_type (type, optional): The type of elements to create. Defaults to SecretInteger.

        Returns:
            NadaArray: The created random NadaArray.

        Raises:
            ValueError: Raised if the nada_type is not supported.
        """
        generator = None
        if nada_type is SecretRational:
            generator = lambda name, party: SecretRational(
                SecretInteger.random(), get_log_scale(), is_scaled=False
            )
        elif nada_type in (SecretInteger, SecretUnsignedInteger):
            generator = lambda name, party: nada_type.random()
        else:
            raise ValueError(f"Unsupported nada_type: {nada_type}")

        return NadaArray(np.array(NadaArray.create_list(dims, None, None, generator)))

    def __getattr__(self, name: str) -> Any:
        """
        Routes other attributes to the inner NumPy array.

        Args:
            name (str): Attribute name.

        Raises:
            AttributeError: Raised if attribute not supported.

        Returns:
            Any: Result of attribute.
        """
        if name not in self.SUPPORTED_OPERATIONS:
            raise AttributeError(
                "NumPy method `%s` is not (currently) supported by NadaArrays." % name
            )

        attr = getattr(self.inner, name)

        if callable(attr):

            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, np.ndarray):
                    return NadaArray(result)
                return result

            return wrapper

        if isinstance(attr, np.ndarray):
            attr = NadaArray(attr)

        return attr

    def __setattr__(self, name: str, value: Any):
        """
        Overrides the default behavior of setting attributes.

        If the attribute name is "inner", it sets the attribute value directly.
        Otherwise, it sets the attribute value on the inner object.

        Args:
            name (str): The name of the attribute.
            value: The value to set for the attribute.
        """
        if name == "inner":
            super().__setattr__(name, value)
        else:
            setattr(self.inner, name, value)

    def __len__(self):
        """
        Overrides the default behavior of returning the length of the object.

        Returns:
            int: The length of the inner numpy array.
        """
        return len(self.inner)

    @property
    def ndim(self) -> int:
        """
        Number of dimensions that the NadaArray holds.

        Returns:
            int: Number of dimenions.
        """
        return len(self.shape)

    @property
    def empty(self) -> bool:
        """
        Whether or not the NadaArray contains any elements.

        Returns:
            bool: Bool result.
        """
        return len(self.inner) == 0

    @property
    def dtype(self) -> Type:
        """
        Gets inner data type of NadaArray values.

        Returns:
            Type: Inner data type.
        """
        if self.empty:
            return NoneType
        return type(self.inner.item(0))

    @property
    def is_rational(self) -> bool:
        """
        Returns whether or not the Array's type is a rational.

        Returns:
            bool: Boolean output.
        """
        return self.dtype in (Rational, SecretRational)

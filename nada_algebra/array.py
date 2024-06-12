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
    rational,
    public_rational,
    secret_rational,
    get_log_scale,
)

from nada_algebra.context import UnsafeArithmeticSession

from nada_algebra.utils import copy_metadata


@dataclass
class NadaArray:
    """
    Represents an array-like object with additional functionality.

    Attributes:
        inner (np.ndarray): The underlying NumPy array.
    """

    def __init__(self, inner: np.ndarray):
        """
        Initializes a new NadaArray object.

        Args:
            inner (np.ndarray): The underlying NumPy array.

        Raises:
            ValueError: Raised if the inner object is not a NumPy array.
        """
        if not isinstance(inner, (np.ndarray, NadaArray)):
            raise ValueError(f"inner must be a numpy array and is: {type(inner)}")
        if isinstance(inner, NadaArray):
            inner = inner.inner
        self.inner = inner

    def __getitem__(self, item):
        """
        Get an item from the array.

        Args:
            item: The item to retrieve.

        Returns:
            NadaArray: A new NadaArray representing the retrieved item.
        """
        ret = self.inner[item]
        if isinstance(ret, np.ndarray):
            return NadaArray(ret)
        return ret

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
            other = other.inner

        ret = self.inner + other

        if isinstance(ret, np.ndarray):
            return NadaArray(ret)
        return ret

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

    def __radd__(self, other: Any) -> "NadaArray":
        """
        Perform element-wise addition with broadcasting.

        Args:
            other (Any): The object to add.

        Returns:
            NadaArray: A new NadaArray representing the element-wise addition result.
        """
        other = NadaArray(np.array(other))
        return other.add(self)

    def sub(self, other: Any) -> "NadaArray":
        """
        Perform element-wise subtraction with broadcasting.

        Args:
            other (Any): The object to subtract.

        Returns:
            NadaArray: A new NadaArray representing the element-wise subtraction result.
        """
        if isinstance(other, NadaArray):
            other = other.inner

        ret = self.inner - other

        if isinstance(ret, np.ndarray):
            return NadaArray(ret)
        return ret

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

    def __rsub__(self, other: Any) -> "NadaArray":
        """
        Perform element-wise subtraction with broadcasting.

        Args:
            other (Any): The object to add.

        Returns:
            NadaArray: A new NadaArray representing the element-wise subtraction result.
        """
        other = NadaArray(np.array(other))
        return other.sub(self)

    def mul(self, other: Any) -> "NadaArray":
        """
        Perform element-wise multiplication with broadcasting.

        Args:
            other (Any): The object to multiply.

        Returns:
            NadaArray: A new NadaArray representing the element-wise multiplication result.
        """
        if isinstance(other, NadaArray):
            other = other.inner

        ret = self.inner * other

        if isinstance(ret, np.ndarray):
            return NadaArray(ret)
        return ret

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

    def __rmul__(self, other: Any) -> "NadaArray":
        """
        Perform element-wise multiplication with broadcasting.

        Args:
            other (Any): The object to add.

        Returns:
            NadaArray: A new NadaArray representing the element-wise multiplication result.
        """
        other = NadaArray(np.array(other))
        return other.mul(self)

    def __neg__(self) -> "NadaArray":
        """
        Performs negation operation.

        Returns:
            NadaArray: Negated NadaArray.
        """
        if self.is_rational:
            return self.apply(lambda x: x * rational(-1))
        return self.apply(lambda x: x * Integer(-1))

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
            other = other.inner

        ret = self.inner / other

        if isinstance(ret, np.ndarray):
            return NadaArray(ret)
        return ret

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

    def __rtruediv__(self, other: Any) -> "NadaArray":
        """
        Perform element-wise division with broadcasting.

        Args:
            other (Any): The object to add.

        Returns:
            NadaArray: A new NadaArray representing the element-wise division result.
        """
        other = NadaArray(np.array(other))
        return other.divide(self)

    def matmul(self, other: "NadaArray") -> "NadaArray":
        """
        Perform matrix multiplication with another NadaArray.

        Args:
            other (NadaArray): The NadaArray to perform matrix multiplication with.

        Returns:
            NadaArray: A new NadaArray representing the result of matrix multiplication.

        Raises:
            TypeError: Raised if the other object is not a NadaArray.
        """

        if not isinstance(other, NadaArray):
            raise TypeError(f"other must be a NadaArray and is: {type(other)}")

        if self.is_rational or other.is_rational:
            return self.rational_matmul(other)

        res = self.inner @ other.inner

        if isinstance(res, np.ndarray):
            return NadaArray(res)
        return res

    def rational_matmul(self, other: "NadaArray") -> "NadaArray":
        """
        Perform matrix multiplication with another NadaArray when both have Rational Numbers.
        It improves the number of truncations to be needed to the resulting matrix dimensions mxp.

        Args:
            other (NadaArray): The NadaArray to perform matrix multiplication with.

        Returns:
            NadaArray: A new NadaArray representing the result of matrix multiplication.
        """
        with UnsafeArithmeticSession():
            res = self.inner @ other.inner

            if isinstance(res, np.ndarray):
                return NadaArray(res).apply(lambda x: x.rescale_down())
            return res.rescale_down()

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
        if not isinstance(other, NadaArray):
            raise TypeError(f"other must be a NadaArray and is: {type(other)}")

        if self.is_rational or other.is_rational:
            return self.rational_matmul(other)

        result = self.inner.dot(other.inner)

        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

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
        res = np.frompyfunc(func, 1, 1)(self.inner)

        if isinstance(res, np.ndarray):
            return NadaArray(res)
        return res

    @copy_metadata(np.ndarray.mean)
    def mean(self, axis=None, dtype=None, out=None, keepdims=False) -> Any:
        sum_arr = self.inner.sum(axis=axis, dtype=dtype, keepdims=keepdims)

        if self.dtype in (Rational, SecretRational):
            nada_type = rational
        else:
            nada_type = Integer

        if axis is None:
            count = nada_type(self.size)
        else:
            if keepdims:
                count = np.expand_dims(count, axis=axis)
                count = np.frompyfunc(nada_type, 1, 1)(count)
            else:
                count = nada_type(self.shape[axis])

        mean_arr = sum_arr / count

        if out is not None:
            out[...] = mean_arr
            return out

        if isinstance(mean_arr, np.ndarray):
            return NadaArray(mean_arr)
        return mean_arr

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

        if len(array.shape) == 0:
            # For compatibility we're leaving this here.
            return NadaArray.output_array(array.item(), party, prefix)

        elif len(array.shape) == 1:
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

    def __len__(self):
        """
        Overrides the default behavior of returning the length of the object.

        Returns:
            int: The length of the inner numpy array.
        """
        return len(self.inner)

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
        # TODO: account for mixed typed NadaArrays due to e.g. padding
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

    def __str__(self) -> str:
        """
        String representation of the NadaArray.

        Returns:
            str: String representation.
        """
        return f"\nNadaArray({self.shape}):\n" + self.debug(self.inner)

    @staticmethod
    def debug(array: np.ndarray) -> str:
        """
        Debug representation of the NadaArray.

        Args:
            array (np.ndarray): The input array.

        Returns:
            str: A string representing the type and shape of the array elements.
        """
        type_mapping = {
            SecretInteger: "(SI)",
            SecretUnsignedInteger: "(SUI)",
            Integer: "(I)",
            UnsignedInteger: "(UI)",
            PublicInteger: "(PI)",
            PublicUnsignedInteger: "(PUI)",
            Rational: "(R)",
            SecretRational: "(SR)",
        }

        # Check for specific types
        for type_class, type_str in type_mapping.items():
            if isinstance(array, type_class):
                return type_str

        # Handle different shapes
        if len(array.shape) == 0:
            return NadaArray.debug(array.item())
        elif len(array.shape) == 1:
            return (
                "["
                + ", ".join(NadaArray.debug(array[i]) for i in range(array.shape[0]))
                + "]"
            )
        return (
            "["
            + "\n".join(NadaArray.debug(array[i]) for i in range(array.shape[0]))
            + "]"
        )

    @copy_metadata(np.ndarray.compress)
    def compress(self, *args, **kwargs):
        result = self.inner.compress(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.copy)
    def copy(self, *args, **kwargs):
        result = self.inner.copy(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.cumprod)
    def cumprod(self, *args, **kwargs):
        result = self.inner.cumprod(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.cumsum)
    def cumsum(self, *args, **kwargs):
        result = self.inner.cumsum(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.diagonal)
    def diagonal(self, *args, **kwargs):
        result = self.inner.diagonal(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.fill)
    def fill(self, *args, **kwargs):
        result = self.inner.fill(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.flatten)
    def flatten(self, *args, **kwargs):
        result = self.inner.flatten(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.item)
    def item(self, *args, **kwargs):
        result = self.inner.item(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.itemset)
    def itemset(self, *args, **kwargs):
        result = self.inner.itemset(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.prod)
    def prod(self, *args, **kwargs):
        result = self.inner.prod(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.put)
    def put(self, *args, **kwargs):
        result = self.inner.put(*args, **kwargs)
        return result

    @copy_metadata(np.ndarray.ravel)
    def ravel(self, *args, **kwargs):
        result = self.inner.ravel(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.repeat)
    def repeat(self, *args, **kwargs):
        result = self.inner.repeat(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.reshape)
    def reshape(self, *args, **kwargs):
        result = self.inner.reshape(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.resize)
    def resize(self, *args, **kwargs):
        result = self.inner.resize(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.squeeze)
    def squeeze(self, *args, **kwargs):
        result = self.inner.squeeze(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.sum)
    def sum(self, *args, **kwargs):
        result = self.inner.sum(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.swapaxes)
    def swapaxes(self, *args, **kwargs):
        result = self.inner.swapaxes(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.take)
    def take(self, *args, **kwargs):
        result = self.inner.take(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.tolist)
    def tolist(self, *args, **kwargs):
        result = self.inner.tolist(*args, **kwargs)
        return result

    @copy_metadata(np.ndarray.trace)
    def trace(self, *args, **kwargs):
        result = self.inner.trace(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @copy_metadata(np.ndarray.transpose)
    def transpose(self, *args, **kwargs):
        result = self.inner.transpose(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @property
    @copy_metadata(np.ndarray.base)
    def base(self):
        result = self.inner.base
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @property
    @copy_metadata(np.ndarray.data)
    def data(self):
        result = self.inner.data
        return result

    @property
    @copy_metadata(np.ndarray.flags)
    def flags(self):
        result = self.inner.flags
        return result

    @property
    @copy_metadata(np.ndarray.flat)
    def flat(self):
        result = self.inner.flat
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @property
    @copy_metadata(np.ndarray.itemsize)
    def itemsize(self):
        result = self.inner.itemsize
        return result

    @property
    @copy_metadata(np.ndarray.nbytes)
    def nbytes(self):
        result = self.inner.nbytes
        return result

    @property
    @copy_metadata(np.ndarray.ndim)
    def ndim(self):
        result = self.inner.ndim
        return result

    @property
    @copy_metadata(np.ndarray.shape)
    def shape(self):
        result = self.inner.shape
        return result

    @property
    @copy_metadata(np.ndarray.size)
    def size(self):
        result = self.inner.size
        return result

    @property
    @copy_metadata(np.ndarray.strides)
    def strides(self):
        result = self.inner.strides
        return result

    @property
    @copy_metadata(np.ndarray.T)
    def T(self):
        result = self.inner.T
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

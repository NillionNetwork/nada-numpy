"""
    This module provides an enhanced array class, NadaArray, with additional functionality
    for mathematical operations and integration with the Nada Numpy library.
"""

# pylint:disable=too-many-lines

from typing import Any, Callable, Optional, Sequence, Union, get_args, overload

import numpy as np
from nada_dsl import (Boolean, Input, Integer, Output, Party, PublicInteger,
                      PublicUnsignedInteger, SecretInteger,
                      SecretUnsignedInteger, UnsignedInteger)

from nada_numpy.context import UnsafeArithmeticSession
from nada_numpy.nada_typing import (NadaBoolean, NadaCleartextType,
                                    NadaInteger, NadaRational,
                                    NadaUnsignedInteger)
from nada_numpy.types import (Rational, SecretRational, get_log_scale,
                              public_rational, rational, secret_rational)
from nada_numpy.utils import copy_metadata


class NadaArray:  # pylint:disable=too-many-public-methods
    """
    Represents an array-like object with additional functionality.
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
        _check_type_conflicts(inner)
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

        Raises:
            ValueError: Raised when value with incompatible type is passed.
        """
        _check_type_compatibility(value, self.dtype)
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

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.mean)
    def mean(self, axis=None, dtype=None, out=None) -> Any:
        sum_arr = self.inner.sum(axis=axis, dtype=dtype)

        if self.is_rational:
            nada_type = rational
        else:
            nada_type = Integer

        if axis is None:
            count = nada_type(self.size)
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
    def _output_array(array: Any, party: Party, prefix: str) -> list:
        """
        Generate a list of Output objects for each element in the input array.

        Args:
            array (Any): The input array.
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
        if isinstance(array, (Rational, SecretRational)):
            return [Output(array.value, f"{prefix}_0", party)]

        if len(array.shape) == 0:
            # For compatibility we're leaving this here.
            return NadaArray._output_array(array.item(), party, prefix)
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
            for v in NadaArray._output_array(array[i], party, f"{prefix}_{i}")
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
        return NadaArray._output_array(self.inner, party, prefix)

    @staticmethod
    def _create_list(
        dims: Sequence[int],
        party: Optional[Party],
        prefix: Optional[str],
        generator: Callable[[Any, Any], Any],
    ) -> list:
        """
        Recursively create a nested list of objects generated by a generator function.

        Args:
            dims (Sequence[int]): A list of integers representing the dimensions
                of the desired nested list.
            party (Party): The party object representing the party to which the elements belong.
            prefix (str): A prefix for the names of the elements.
            generator (Callable[[Any, Any], Any]): The function to generate the elements.

        Returns:
            list: A nested list of generated objects.
        """
        if len(dims) == 1:
            return [generator(f"{prefix}_{i}", party) for i in range(dims[0])]
        return [
            NadaArray._create_list(
                dims[1:],
                party,
                f"{prefix}_{i}",
                generator,
            )
            for i in range(dims[0])
        ]

    @staticmethod
    def array(
        dims: Sequence[int],
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
            dims (Sequence[int]): A list of integers representing the dimensions of the array.
            party (Party): The party object.
            prefix (str): A prefix for naming the array elements.
            nada_type (type): The type of elements to create.

        Returns:
            NadaArray: The created NadaArray.

        Raises:
            ValueError: Raised if the nada_type is not supported.
        """
        generator: Callable[[Any, Any], Any]
        if nada_type == Rational:
            # pylint:disable=unnecessary-lambda-assignment
            generator = lambda name, party: public_rational(name=name, party=party)
        elif nada_type == SecretRational:
            # pylint:disable=unnecessary-lambda-assignment
            generator = lambda name, party: secret_rational(name=name, party=party)
        elif nada_type in (
            SecretInteger,
            SecretUnsignedInteger,
            PublicInteger,
            PublicUnsignedInteger,
        ):
            # pylint:disable=unnecessary-lambda-assignment
            generator = lambda name, party: nada_type(Input(name=name, party=party))  # type: ignore
        else:
            raise ValueError(f"Unsupported nada_type: {nada_type}")

        return NadaArray(
            np.array(NadaArray._create_list(dims, party, prefix, generator))
        )

    @staticmethod
    def random(
        dims: Sequence[int],
        nada_type: Union[
            SecretInteger, SecretUnsignedInteger, SecretRational
        ] = SecretInteger,
    ) -> "NadaArray":
        """
        Create a random NadaArray with the specified dimensions.

        Args:
            dims (Sequence[int]): A list of integers representing the dimensions of the array.
            nada_type (type, optional): The type of elements to create. Defaults to SecretInteger.

        Returns:
            NadaArray: The created random NadaArray.

        Raises:
            ValueError: Raised if the nada_type is not supported.
        """
        if nada_type is SecretRational:
            # pylint:disable=unnecessary-lambda-assignment
            generator = lambda name, party: SecretRational(
                SecretInteger.random(), get_log_scale(), is_scaled=False
            )
        elif nada_type == SecretInteger:
            # pylint:disable=unnecessary-lambda-assignment
            generator = lambda name, party: SecretInteger.random()
        elif nada_type == SecretUnsignedInteger:
            # pylint:disable=unnecessary-lambda-assignment
            generator = lambda name, party: SecretUnsignedInteger.random()
        else:
            raise ValueError(f"Unsupported nada_type: {nada_type}")

        return NadaArray(np.array(NadaArray._create_list(dims, None, None, generator)))

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
    def dtype(
        self,
    ) -> Optional[Union[NadaRational, NadaInteger, NadaUnsignedInteger, NadaBoolean]]:
        """
        Gets data type of array.

        Returns:
            Optional[
                Union[NadaRational, NadaInteger, NadaUnsignedInteger, NadaBoolean]
            ]: Array data type if applicable.
        """
        return get_dtype(self.inner)

    @property
    def is_rational(self) -> bool:
        """
        Returns whether or not the Array type contains rationals.

        Returns:
            bool: Boolean output.
        """
        return self.dtype == NadaRational

    @property
    def is_integer(self) -> bool:
        """
        Returns whether or not the Array type contains signed integers.

        Returns:
            bool: Boolean output.
        """
        return self.dtype == NadaInteger

    @property
    def is_unsigned_integer(self) -> bool:
        """
        Returns whether or not the Array type contains unsigned integers.

        Returns:
            bool: Boolean output.
        """
        return self.dtype == NadaUnsignedInteger

    @property
    def is_boolean(self) -> bool:
        """
        Returns whether or not the Array type contains signed integers.

        Returns:
            bool: Boolean output.
        """
        return self.dtype == NadaBoolean

    @property
    def cleartext_nada_type(self) -> NadaCleartextType:
        """
        Returns a clear-text Nada type compatible with the Nada array.

        Returns:
            NadaCleartextType: Compatible cleartext type.
        """
        if self.is_rational:
            return Rational
        if self.is_integer:
            return Integer
        if self.is_unsigned_integer:
            return UnsignedInteger
        if self.is_boolean:
            return Boolean
        raise TypeError(f"Array {self} is of unknown type {self.dtype}.")

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
        if len(array.shape) == 1:
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

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.compress)
    def compress(self, *args, **kwargs):
        result = self.inner.compress(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.copy)
    def copy(self, *args, **kwargs):
        result = self.inner.copy(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.cumprod)
    def cumprod(self, *args, **kwargs):
        result = self.inner.cumprod(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.cumsum)
    def cumsum(self, *args, **kwargs):
        result = self.inner.cumsum(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.diagonal)
    def diagonal(self, *args, **kwargs):
        result = self.inner.diagonal(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.fill)
    def fill(self, *args, **kwargs):
        result = self.inner.fill(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.flatten)
    def flatten(self, *args, **kwargs):
        result = self.inner.flatten(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.item)
    def item(self, *args, **kwargs):
        result = self.inner.item(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @overload
    def itemset(self, value: Any): ...
    @overload
    def itemset(self, item: Any, value: Any): ...

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.itemset)
    def itemset(self, *args, **kwargs):
        value = None
        if len(args) == 1:
            value = args[0]
        elif len(args) == 2:
            value = args[1]
        else:
            value = kwargs["value"]

        _check_type_compatibility(value, self.dtype)
        result = self.inner.itemset(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.prod)
    def prod(self, *args, **kwargs):
        result = self.inner.prod(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.put)
    def put(self, ind: Any, v: Any, mode: Any = None) -> None:
        _check_type_compatibility(v, self.dtype)
        if isinstance(v, NadaArray):
            self.inner.put(ind, v.inner, mode)
        else:
            self.inner.put(ind, v, mode)

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.ravel)
    def ravel(self, *args, **kwargs):
        result = self.inner.ravel(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.repeat)
    def repeat(self, *args, **kwargs):
        result = self.inner.repeat(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.reshape)
    def reshape(self, *args, **kwargs):
        result = self.inner.reshape(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.resize)
    def resize(self, *args, **kwargs):
        result = self.inner.resize(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.squeeze)
    def squeeze(self, *args, **kwargs):
        result = self.inner.squeeze(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.sum)
    def sum(self, *args, **kwargs):
        result = self.inner.sum(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.swapaxes)
    def swapaxes(self, *args, **kwargs):
        result = self.inner.swapaxes(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.take)
    def take(self, *args, **kwargs):
        result = self.inner.take(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.tolist)
    def tolist(self, *args, **kwargs):
        result = self.inner.tolist(*args, **kwargs)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.trace)
    def trace(self, *args, **kwargs):
        result = self.inner.trace(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    # pylint:disable=missing-function-docstring
    @copy_metadata(np.ndarray.transpose)
    def transpose(self, *args, **kwargs):
        result = self.inner.transpose(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @property
    # pylint:disable=missing-function-docstring,no-member
    @copy_metadata(np.ndarray.base)
    def base(self):
        result = self.inner.base
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @property
    # pylint:disable=missing-function-docstring,no-member
    @copy_metadata(np.ndarray.data)
    def data(self):
        result = self.inner.data
        return result

    @property
    # pylint:disable=missing-function-docstring,no-member
    @copy_metadata(np.ndarray.flags)
    def flags(self):
        result = self.inner.flags
        return result

    @property
    # pylint:disable=missing-function-docstring,no-member
    @copy_metadata(np.ndarray.flat)
    def flat(self):
        result = self.inner.flat
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result

    @property
    # pylint:disable=missing-function-docstring,no-member
    @copy_metadata(np.ndarray.itemsize)
    def itemsize(self):
        result = self.inner.itemsize
        return result

    @property
    # pylint:disable=missing-function-docstring,no-member
    @copy_metadata(np.ndarray.nbytes)
    def nbytes(self):
        result = self.inner.nbytes
        return result

    @property
    # pylint:disable=missing-function-docstring,no-member
    @copy_metadata(np.ndarray.ndim)
    def ndim(self):
        result = self.inner.ndim
        return result

    @property
    # pylint:disable=missing-function-docstring,no-member
    @copy_metadata(np.ndarray.shape)
    def shape(self):
        result = self.inner.shape
        return result

    @property
    # pylint:disable=missing-function-docstring,no-member
    @copy_metadata(np.ndarray.size)
    def size(self):
        result = self.inner.size
        return result

    @property
    # pylint:disable=missing-function-docstring,no-member
    @copy_metadata(np.ndarray.strides)
    def strides(self):
        result = self.inner.strides
        return result

    @property
    # pylint:disable=missing-function-docstring,no-member
    @copy_metadata(np.ndarray.T)
    def T(self):  # pylint:disable=invalid-name
        result = self.inner.T
        if isinstance(result, np.ndarray):
            return NadaArray(result)
        return result


def _check_type_compatibility(
    value: Any,
    check_type: Optional[
        Union[NadaRational, NadaInteger, NadaUnsignedInteger, NadaBoolean]
    ],
) -> None:
    """
    Checks type compatibility between a type and a Nada base type.

    Args:
        value (Any): Value to be type-checked.
        check_type (Optional[
            Union[NadaRational, NadaInteger, NadaUnsignedInteger, NadaBoolean]
        ]): Base Nada type to check against.

    Raises:
        TypeError: Raised when types are not compatible.
    """
    if isinstance(value, (NadaArray, np.ndarray)):
        if isinstance(value, NadaArray):
            value = value.inner
        dtype = get_dtype(value)
        if dtype is None or check_type is None:
            raise TypeError(f"Type {dtype} is not compatible with {check_type}")
        if dtype == check_type:
            return
    else:
        dtype = type(value)
        if dtype in get_args(check_type):
            return
    raise TypeError(f"Type {dtype} is not compatible with {check_type}")


def _check_type_conflicts(array: np.ndarray) -> None:
    """
    Checks for type conflicts

    Args:
        array (np.ndarray): Array to be checked.

    Raises:
        TypeError: Raised when incompatible dtypes are detected.
    """
    _ = get_dtype(array)


def get_dtype(
    array: np.ndarray,
) -> Optional[Union[NadaRational, NadaInteger, NadaUnsignedInteger, NadaBoolean]]:
    """
    Gets all data types present in array.

    Args:
        array (np.ndarray): Array to be checked.

    Raises:
        TypeError: Raised when incompatible dtypes are detected.

    Returns:
        Optional[Union[NadaRational, NadaInteger, NadaUnsignedInteger, NadaBoolean]: Array dtype].
    """
    if array.size == 0:
        return None

    unique_types = set(type(element) for element in array.flat)

    base_dtypes = [NadaRational, NadaInteger, NadaUnsignedInteger, NadaBoolean]
    for base_dtype in base_dtypes:
        if all(unique_type in get_args(base_dtype) for unique_type in unique_types):
            return base_dtype
    raise TypeError(f"Nada-incompatible dtypes detected in `{unique_types}`.")

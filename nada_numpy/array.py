"""
    This module provides an enhanced array class, NadaArray, with additional functionality
    for mathematical operations and integration with the Nada Numpy library.
"""

# pylint:disable=too-many-lines

from typing import (Any, Callable, Optional, Sequence, Tuple, Union, get_args,
                    overload)

import numpy as np
from nada_dsl import (Boolean, Input, Integer, Output, Party, PublicInteger,
                      PublicUnsignedInteger, SecretInteger,
                      SecretUnsignedInteger, UnsignedInteger)

from nada_numpy.context import UnsafeArithmeticSession
from nada_numpy.nada_typing import (AnyNadaType, NadaBoolean,
                                    NadaCleartextType, NadaInteger,
                                    NadaRational, NadaUnsignedInteger)
from nada_numpy.types import (Rational, SecretBoolean, SecretRational, fxp_abs,
                              get_log_scale, public_rational, rational,
                              secret_rational, sign)
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

    def __comparison_operator(
        self, value: Union["NadaArray", "AnyNadaType", np.ndarray], operator: Callable
    ) -> "NadaArray":
        """
        Perform element-wise comparison with broadcasting.

        NOTE: Specially for __eq__ and __ne__ operators, the result expected is bool.
        If we don't define this method, the result will be a NadaArray with bool outputs.

        Args:
            value (Any): The object to compare.
            operator (str): The comparison operator.

        Returns:
            NadaArray: A new NadaArray representing the element-wise comparison result.
        """
        if isinstance(value, NadaArray):
            value = value.inner
        if isinstance(
            value,
            (
                SecretInteger,
                Integer,
                SecretUnsignedInteger,
                UnsignedInteger,
                SecretRational,
                Rational,
            ),
        ):
            return self.apply(lambda x: operator(x, value))

        if isinstance(value, np.ndarray):
            if len(self.inner) != len(value):
                raise ValueError("Arrays must have the same length")
            vectorized_operator = np.vectorize(operator)
            return NadaArray(vectorized_operator(self.inner, value))

        raise ValueError(f"Unsupported type: {type(value)}")

    def __eq__(self, value: Any) -> "NadaArray":  # type: ignore
        """
        Perform equality comparison with broadcasting.

        Args:
            value (object): The object to compare.

        Returns:
            NadaArray: A boolean representing the element-wise equality comparison result.
        """
        return self.__comparison_operator(value, lambda x, y: x == y)

    def __ne__(self, value: Any) -> "NadaArray":  # type: ignore
        """
        Perform inequality comparison with broadcasting.

        Args:
            value (object): The object to compare.

        Returns:
            NadaArray: A boolean array representing the element-wise inequality comparison result.
        """
        return self.__comparison_operator(value, lambda x, y: ~(x == y))

    def __lt__(self, value: Any) -> "NadaArray":
        """
        Perform less than comparison with broadcasting.

        Args:
            value (object): The object to compare.

        Returns:
            NadaArray: A boolean array representing the element-wise less than comparison result.
        """
        return self.__comparison_operator(value, lambda x, y: x < y)

    def __le__(self, value: Any) -> "NadaArray":
        """
        Perform less than or equal comparison with broadcasting.

        Args:
            value (object): The object to compare.

        Returns:
            NadaArray: A boolean array representing
                the element-wise less or equal thancomparison result.
        """
        return self.__comparison_operator(value, lambda x, y: x <= y)

    def __gt__(self, value: Any) -> "NadaArray":
        """
        Perform greater than comparison with broadcasting.

        Args:
            value (object): The object to compare.

        Returns:
            NadaArray: A boolean array representing the element-wise greater than comparison result.
        """
        return self.__comparison_operator(value, lambda x, y: x > y)

    def __ge__(self, value: Any) -> "NadaArray":
        """
        Perform greater than or equal comparison with broadcasting.

        Args:
            value (object): The object to compare.

        Returns:
            NadaArray: A boolean representing the element-wise greater or equal than comparison.
        """
        return self.__comparison_operator(value, lambda x, y: x >= y)

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

    def shuffle(self) -> "NadaArray":
        """
        Shuffles a 1D array using the Benes network.

        This function rearranges the elements of a 1-dimensional array in a deterministic but
        seemingly random order based on the Benes network, a network used in certain types of
        sorting and switching circuits. The Benes network requires the input array's length
        to be a power of two (e.g., 2, 4, 8, 16, ...).

        Note: The resulting shuffled arrays contain the same elements as the input arrays.

        Args:
            NadaArray: The input array to be shuffled. This must be a 1-dimensional NumPy array.
                The length of the array must be a power of two.

        Returns:
            NadaArray: The shuffled version of the input array. The output is a new array where
                the elements have been rearranged according to the Benes network.

        Raises:
            ValueError: If the length of the input array is not a power of two.

        Example:
        ```python
        import nada_numpy as na

        # Example arrays with different data types
        parties = na.parties(2)
        a = na.array([8], parties[0], "A", na.Rational)
        b = na.array([8], parties[0], "B", na.SecretRational)
        c = na.array([8], parties[0], "C", PublicInteger)
        d = na.array([8], parties[0], "D", SecretInteger)

        # Shuffling the arrays
        shuffled_a = a.shuffle()
        shuffled_b = b.shuffle()
        shuffled_c = c.shuffle()
        ```

        Frequency analysis:

            This script performs a frequency analysis of a shuffle function implemented using a
            Benes network. It includes a function for shuffle, a test function for evaluating
            randomness, and an example of running the test. Below is an overview of the code and
            its output.

            1. **Shuffle Function**:

            The `shuffle` function shuffles a 1D array using a Benes network approach.
            The Benes network is defined by the function `_benes_network(n)`, which should provide
            the network stages required for the shuffle.

            ```python
            import numpy as np
            import random

            def rand_bool():
                # Simulates a random boolean value
                return random.choice([0, 1]) == 0

            def swap_gate(a, b):
                # Conditionally swaps two values based on a random boolean
                rbool = rand_bool()
                return (b, a) if rbool else (a, b)

            def shuffle(array):
                # Applies Benes network shuffle to a 1D array
                if array.ndim != 1:
                    raise ValueError("Input array must be a 1D array.")

                n = array.size
                bnet = benes_network(n)
                swap_array = np.ones(n)

                first_numbers = np.arange(0, n, 2)
                second_numbers = np.arange(1, n, 2)
                pairs = np.column_stack((first_numbers, second_numbers))

                for stage in bnet:
                    for ((i0, i1), (a, b)) in zip(pairs, stage):
                        swap_array[i0], swap_array[i1] = swap_gate(array[a], array[b])
                    array = swap_array.copy()

                return array
                ```

            2. **Randomness Test Function:**:
            The test_shuffle_randomness function evaluates the shuffle function by performing
            multiple shuffles and counting the occurrences of each element at each position.

                    ```python
                    def test_shuffle_randomness(vector_size, num_shuffles):
                        # Initializes vector and count matrix
                        vector = np.arange(vector_size)
                        counts = np.zeros((vector_size, vector_size), dtype=int)

                        # Performs shuffling and counts occurrences
                        for _ in range(num_shuffles):
                            shuffled_vector = shuffle(vector)
                            for position, element in enumerate(shuffled_vector):
                                counts[int(element), position] += 1

                        # Computes average counts and deviation
                        average_counts = num_shuffles / vector_size
                        deviation = np.abs(counts - average_counts)

                        return counts, average_counts, deviation
                    ```


            Running the `test_shuffle_randomness` function with a vector size of 8 and 100,000
            shuffles provides the following results:

                    ```python
                    vector_size = 8  # Size of the vector
                    num_shuffles = 100000  # Number of shuffles to perform

                    counts, average_counts, deviation = test_shuffle_randomness(vector_size,
                                                                                num_shuffles)

                    print("Counts of numbers appearances at each position:")
                    for i in range(vector_size):
                        print(f"Number {i}: {counts[i]}")
                    print("Expected count of number per slot:", average_counts)
                    print("\nDeviation from the expected average:")
                    for i in range(vector_size):
                        print(f"Number {i}: {deviation[i]}")
                    ```
                    ```bash
                    >>> Counts of numbers appearances at each position:
                    >>> Number 0: [12477 12409 12611 12549 12361 12548 12591 12454]
                    >>> Number 1: [12506 12669 12562 12414 12311 12408 12377 12753]
                    >>> Number 2: [12595 12327 12461 12607 12492 12721 12419 12378]
                    >>> Number 3: [12417 12498 12586 12433 12627 12231 12638 12570]
                    >>> Number 4: [12370 12544 12404 12337 12497 12743 12588 12517]
                    >>> Number 5: [12559 12420 12416 12791 12508 12489 12360 12457]
                    >>> Number 6: [12669 12459 12396 12394 12757 12511 12423 12391]
                    >>> Number 7: [12407 12674 12564 12475 12447 12349 12604 12480]
                    >>> Expected count of number per slot: 12500.0
                    >>>
                    >>> Deviation from the expected average:
                    >>> Number 0: [ 23.  91. 111.  49. 139.  48.  91.  46.]
                    >>> Number 1: [  6. 169.  62.  86. 189.  92. 123. 253.]
                    >>> Number 2: [ 95. 173.  39. 107.   8. 221.  81. 122.]
                    >>> Number 3: [ 83.   2.  86.  67. 127. 269. 138.  70.]
                    >>> Number 4: [130.  44.  96. 163.   3. 243.  88.  17.]
                    >>> Number 5: [ 59.  80.  84. 291.   8.  11. 140.  43.]
                    >>> Number 6: [169.  41. 104. 106. 257.  11.  77. 109.]
                    >>> Number 7: [ 93. 174.  64.  25.  53. 151. 104.  20.]
                    ```
        """
        arr = self.copy()
        # Ensure the array is a 1D array
        if arr.ndim != 1:
            raise ValueError("Input array must be a 1D array.")

        n = arr.size
        bnet = _benes_network(n)
        swap_arr = arr.copy()

        evens = np.arange(0, n, 2)
        odds = np.arange(1, n, 2)
        pairs = np.column_stack((evens, odds))
        for stage in bnet:
            for (i0, i1), (a, b) in zip(pairs, stage):
                swap_arr[i0], swap_arr[i1] = _swap_gate(arr[a], arr[b])
            arr = swap_arr.copy()

        return arr

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

    # Non-linear functions

    def sign(self) -> "NadaArray":
        """Computes the sign value (0 is considered positive)"""
        if self.is_rational:
            return self.apply(sign)

        dtype = get_dtype(self.inner)
        raise TypeError(
            f"Sign is not compatible with {dtype}, only with Rational and SecretRational types."
        )

    def abs(self) -> "NadaArray":
        """Computes the absolute value"""
        if self.is_rational:
            return self.apply(fxp_abs)

        dtype = get_dtype(self.inner)
        raise TypeError(
            f"Abs is not compatible with {dtype}, only with Rational and SecretRational types."
        )

    def exp(self, iterations: int = 8) -> "NadaArray":
        """
        Approximates the exponential function using a limit approximation.

        The exponential function is approximated using the following limit:

            exp(x) = lim_{n -> ∞} (1 + x / n) ^ n

        The exponential function is computed by choosing n = 2 ** d, where d is set to `iterations`.
        The calculation is performed by computing (1 + x / n) once and then squaring it `d` times.

        Approximation accuracy range (with 16 bit precision):
        + ---------------------------------- +
        |  Input range x |  Relative error   |
        + ---------------------------------- +
        |   [-2, 2]      |       <1%         |
        |   [-7, 7]      |       <10%        |
        |   [-8, 15]     |       <35%        |
        + ---------------------------------- +

        Args:
            iterations (int, optional): The number of iterations for the limit approximation.
                Defaults to 8.

        Returns:
            NadaArray: The approximated value of the exponential function.
        """
        if self.is_rational:

            def exp(x):
                return x.exp(iterations=iterations)

            return self.apply(exp)

        dtype = get_dtype(self.inner)
        raise TypeError(
            f"Exp is not compatible with {dtype}, only with Rational and SecretRational types."
        )

    def polynomial(self, coefficients: list) -> "NadaArray":
        """
        Computes a polynomial function on a value with given coefficients.

        The coefficients can be provided as a list of values.
        They should be ordered from the linear term (order 1) first,
        ending with the highest order term.
        **Note: The constant term is not included.**

        Args:
            coefficients (list): The coefficients of the polynomial, ordered by increasing degree.

        Returns:
            NadaArray: The result of the polynomial function applied to the input x.
        """
        if self.is_rational:

            def polynomial(x):
                return x.polynomial(coefficients=coefficients)

            return self.apply(polynomial)

        dtype = get_dtype(self.inner)
        raise TypeError(
            f"Polynomial is not compatible with {dtype},\
                only with Rational and SecretRational types."
        )

    def log(
        self,
        input_in_01: bool = False,
        iterations: int = 2,
        exp_iterations: int = 8,
        order: int = 8,
    ) -> "NadaArray":
        """
        Approximates the natural logarithm using 8th order modified Householder iterations.
        This approximation is accurate within 2% relative error on the interval [0.0001, 250].

        The iterations are computed as follows:

            h = 1 - x * exp(-y_n)
            y_{n+1} = y_n - sum(h^k / k for k in range(1, order + 1))

        Approximation accuracy range (with 16 bit precision):
        + ------------------------------------- +
        |    Input range x  |  Relative error   |
        + ------------------------------------- +
        | [0.001, 200]      |     <1%           |
        | [0.00003, 253]    |     <10%          |
        | [0.0000001, 253]  |     <40%          |
        | [253, +∞[         |     Unstable      |
        + ------------------------------------- +

        Args:
            input_in_01 (bool, optional): Indicates if the input is within the domain [0, 1].
                This setting optimizes the function for this domain, useful for computing
                log-probabilities in entropy functions.

                To shift the domain of convergence, a constant 'a' is used with the identity:

                    ln(u) = ln(au) - ln(a)

                Given the convergence domain for log() function is approximately [1e-4, 1e2],
                we set a = 100.
                Defaults to False.
            iterations (int, optional): Number of Householder iterations for the approximation.
                Defaults to 2.
            exp_iterations (int, optional): Number of iterations for the limit
                approximation of exp. Defaults to 8.
            order (int, optional): Number of polynomial terms used (order of
                Householder approximation). Defaults to 8.

        Returns:
            NadaArray: The approximate value of the natural logarithm.
        """
        if self.is_rational:

            def log(x):
                return x.log(
                    input_in_01=input_in_01,
                    iterations=iterations,
                    exp_iterations=exp_iterations,
                    order=order,
                )

            return self.apply(log)

        dtype = get_dtype(self.inner)
        raise TypeError(
            f"Log is not compatible with {dtype}, only with Rational and SecretRational types."
        )

    def reciprocal(  # pylint: disable=too-many-arguments
        self,
        all_pos: bool = False,
        initial: Optional["Rational"] = None,
        input_in_01: bool = False,
        iterations: int = 10,
        log_iters: int = 1,
        exp_iters: int = 8,
        method: str = "NR",
    ) -> "NadaArray":
        r"""
        Approximates the reciprocal of a number through two possible methods: Newton-Raphson
        and log.

        Methods:
            'NR' : `Newton-Raphson`_ method computes the reciprocal using iterations
                    of :math:`x_{i+1} = (2x_i - x * x_i^2)` and uses
                    :math:`3*exp(1 - 2x) + 0.003` as an initial guess by default.

                    Approximation accuracy range (with 16 bit precision):
                    + ------------------------------------ +
                    | Input range |x|  |  Relative error   |
                    + ------------------------------------ +
                    | [0.1, 64]        |       <0%         |
                    | [0.0003, 253]    |       <15%        |
                    | [0.00001, 253]   |       <90%        |
                    | [253, +∞[        |     Unstable      |
                    + ------------------------------------ +

            'log' : Computes the reciprocal of the input from the observation that:
                    :math:`x^{-1} = exp(-log(x))`

                    Approximation accuracy range (with 16 bit precision):
                    + ------------------------------------ +
                    | Input range |x|  |  Relative error   |
                    + ------------------------------------ +
                    | [0.0003, 253]    |       <15%        |
                    | [0.00001, 253]   |       <90%        |
                    | [253, +∞[        |     Unstable      |
                    + ------------------------------------ +

        Args:
            all_pos (bool, optional): determines whether all elements of the
                input are known to be positive, which optimizes the step of
                computing the sign of the input. Defaults to False.
            initial (Rational, optional): sets the initial value for the
                Newton-Raphson method. By default, this will be set to :math:
                `3*exp(-(x-.5)) + 0.003` as this allows the method to converge over
                a fairly large domain.
            input_in_01 (bool, optional) : Allows a user to indicate that the input is
                        in the range [0, 1], causing the function optimize for this range.
                        This is useful for improving the accuracy of functions on
                        probabilities (e.g. entropy functions).
            iterations (int, optional):  determines the number of Newton-Raphson iterations to run
                            for the `NR` method. Defaults to 10.
            log_iters (int, optional): determines the number of Householder
                iterations to run when computing logarithms for the `log` method. Defaults to 1.
            exp_iters (int, optional): determines the number of exp
                iterations to run when computing exp. Defaults to 8.
            method (str, optional): method used to compute reciprocal. Defaults to "NR".

        Returns:
            NadaArray: The approximate value of the reciprocal

        .. _Newton-Raphson:
            https://en.wikipedia.org/wiki/Newton%27s_method
        """
        if self.is_rational:
            # pylint:disable=duplicate-code
            def reciprocal(x):
                return x.reciprocal(
                    all_pos=all_pos,
                    initial=initial,
                    input_in_01=input_in_01,
                    iterations=iterations,
                    log_iters=log_iters,
                    exp_iters=exp_iters,
                    method=method,
                )

            return self.apply(reciprocal)

        dtype = get_dtype(self.inner)
        raise TypeError(
            f"Reciprocal is not compatible with {dtype},\
                only with Rational and SecretRational types."
        )

    def inv_sqrt(
        self,
        initial: Optional["SecretRational"] = None,
        iterations: int = 5,
        method: str = "NR",
    ) -> "NadaArray":
        r"""
        Computes the inverse square root of the input using the Newton-Raphson method.

        Approximation accuracy range (with 16 bit precision):
        + ---------------------------------- +
        | Input range x  |  Relative error   |
        + ---------------------------------- +
        | [0.1, 170]     |       <0%         |
        | [0.001, 200]   |       <50%        |
        | [200, +∞[      |     Unstable      |
        + ---------------------------------- +

        Args:
            initial (Union[SecretRational, None], optional): sets the initial value for the
                        Newton-Raphson iterations. By default, this will be set to allow the
                        method to converge over a fairly large domain.
            iterations (int, optional): determines the number of Newton-Raphson iterations to run.
            method (str, optional): method used to compute inv_sqrt. Defaults to "NR".

        Returns:
            NadaArray: The approximate value of the inv_sqrt.

        .. _Newton-Raphson:
            https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
        """
        if self.is_rational:

            def inv_sqrt(x):
                return x.inv_sqrt(initial=initial, iterations=iterations, method=method)

            return self.apply(inv_sqrt)

        dtype = get_dtype(self.inner)
        raise TypeError(
            f"Inverse square-root is not compatible with {dtype},\
                only with Rational and SecretRational types."
        )

    def sqrt(
        self,
        initial: Optional["SecretRational"] = None,
        iterations: int = 5,
        method: str = "NR",
    ) -> "NadaArray":
        r"""
        Computes the square root of the input by computing its inverse square root using
        the Newton-Raphson method and multiplying by the input.

        Approximation accuracy range (with 16 bit precision):
        + ---------------------------------- +
        | Input range x  |  Relative error   |
        + ---------------------------------- +
        | [0.1, 170]     |       <0%         |
        | [0.001, 200]   |       <50%        |
        | [200, +∞[      |     Unstable      |
        + ---------------------------------- +

        Args:
            initial (Union[SecretRational, None], optional): sets the initial value for the inverse
                square root Newton-Raphson iterations. By default, this will be set to allow
                convergence over a fairly large domain. Defaults to None.
            iterations (int, optional):  determines the number of Newton-Raphson iterations to run.
                Defaults to 5.
            method (str, optional): method used to compute sqrt. Defaults to "NR".

        Returns:
            NadaArray: The approximate value of the sqrt.

        .. _Newton-Raphson:
            https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
        """
        if self.is_rational:

            def sqrt(x):
                return x.sqrt(initial=initial, iterations=iterations, method=method)

            return self.apply(sqrt)

        dtype = get_dtype(self.inner)
        raise TypeError(
            f"Square-root is not compatible with {dtype},\
                only with Rational and SecretRational types."
        )

    # Trigonometry

    def cossin(self, iterations: int = 10) -> "NadaArray":
        r"""Computes cosine and sine through e^(i * input) where i is the imaginary unit through the
        formula:

        .. math::
            Re\{e^{i * input}\}, Im\{e^{i * input}\} = \cos(input), \sin(input)

        Args:
            iterations (int, optional): determines the number of iterations to run. Defaults to 10.

        Returns:
            NadaArray:
                An array of tuples where the first element is cos and the second element is the sin.
        """
        if self.is_rational:

            def cossin(x):
                return x.cossin(iterations=iterations)

            return self.apply(cossin)

        dtype = get_dtype(self.inner)
        raise TypeError(
            f"Cosine and Sine are not compatible with {dtype},\
                only with Rational and SecretRational types."
        )

    def cos(self, iterations: int = 10) -> "NadaArray":
        r"""Computes the cosine of the input using cos(x) = Re{exp(i * x)}.

        Note: unstable outside [-30, 30]

        Args:
            iterations (int, optional): determines the number of iterations to run. Defaults to 10.

        Returns:
            NadaArray: The approximate value of the cosine.
        """
        if self.is_rational:

            def cos(x):
                return x.cos(iterations=iterations)

            return self.apply(cos)

        dtype = get_dtype(self.inner)
        raise TypeError(
            f"Cosine is not compatible with {dtype},\
                only with Rational and SecretRational types."
        )

    def sin(self, iterations: int = 10) -> "NadaArray":
        r"""Computes the sine of the input using sin(x) = Im{exp(i * x)}.

        Note: unstable outside [-30, 30]

        Args:
            iterations (int, optional): determines the number of iterations to run. Defaults to 10.

        Returns:
            NadaArray: The approximate value of the sine.
        """
        if self.is_rational:

            def sin(x):
                return x.sin(iterations=iterations)

            return self.apply(sin)

        dtype = get_dtype(self.inner)
        raise TypeError(
            f"Sine is not compatible with {dtype}, only with Rational and SecretRational types."
        )

    def tan(self, iterations: int = 10) -> "NadaArray":
        r"""Computes the tan of the input using tan(x) = sin(x) / cos(x).

        Note: unstable outside [-30, 30]

        Args:
            iterations (int, optional): determines the number of iterations to run. Defaults to 10.

        Returns:
            NadaArray: The approximate value of the tan.
        """
        if self.is_rational:

            def tan(x):
                return x.tan(iterations=iterations)

            return self.apply(tan)

        dtype = get_dtype(self.inner)
        raise TypeError(
            f"Tangent is not compatible with {dtype},\
                only with Rational and SecretRational types."
        )

    # Activation functions

    def tanh(
        self, chebyshev_terms: int = 32, method: str = "reciprocal"
    ) -> "NadaArray":
        r"""Computes the hyperbolic tangent function using the identity

        .. math::
            tanh(x) = 2\sigma(2x) - 1

        Methods:
        If a valid method is given, this function will compute tanh using that method:

            "reciprocal" - computes tanh using the identity

                .. math::
                tanh(x) = 2\sigma(2x) - 1

                Note: stable for x in [-250, 250]. Unstable otherwise.

            "chebyshev" - computes tanh via Chebyshev approximation with truncation.

                .. math::
                    tanh(x) = \sum_{j=1}^chebyshev_terms c_{2j - 1} P_{2j - 1} (x / maxval)

                where c_i is the ith Chebyshev series coefficient and P_i is ith polynomial.

                Note: stable for all input range as the approximation is truncated
                        to +/-1 outside [-1, 1].

            "motzkin" - computes tanh via approximation from the paper
                "BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers"
                on section 5.3 based on the Motzkin’s polynomial preprocessing technique.

                Note: stable for all input range as the approximation is truncated
                        to +/-1 outside [-1, 1].

        Args:
            chebyshev_terms (int, optional): highest degree of Chebyshev polynomials.
                            Must be even and at least 6. Defaults to 32.
            method (str, optional): method used to compute tanh function. Defaults to "reciprocal".

        Returns:
            NadaArray: The tanh evaluation.

        Raises:
            ValueError: Raised if method type is not supported.
        """
        if self.is_rational:

            def tanh(x):
                return x.tanh(chebyshev_terms=chebyshev_terms, method=method)

            return self.apply(tanh)

        dtype = get_dtype(self.inner)
        raise TypeError(
            f"Hyperbolic tangent is not compatible with {dtype},\
                only with Rational and SecretRational types."
        )

    def sigmoid(
        self, chebyshev_terms: int = 32, method: str = "reciprocal"
    ) -> "NadaArray":
        r"""Computes the sigmoid function using the following definition

        .. math::
            \sigma(x) = (1 + e^{-x})^{-1}

        Methods:
        If a valid method is given, this function will compute sigmoid
            using that method:

            "chebyshev" - computes tanh via Chebyshev approximation with
                truncation and uses the identity:

                .. math::
                    \sigma(x) = \frac{1}{2}tanh(\frac{x}{2}) + \frac{1}{2}

                Note: stable for all input range as the approximation is truncated
                        to 0/1 outside [-1, 1].

            "motzkin" - computes tanh via approximation from the paper
                "BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers"
                on section 5.3 based on the Motzkin’s polynomial preprocessing technique. It uses
                the identity:

                .. math::
                    \sigma(x) = \frac{1}{2}tanh(\frac{x}{2}) + \frac{1}{2}

                Note: stable for all input range as the approximation is truncated
                        to 0/1 outside [-1, 1].

            "reciprocal" - computes sigmoid using :math:`1 + e^{-x}` and computing
                the reciprocal

                Note: stable for x in [-500, 500]. Unstable otherwise.

        Args:
            chebyshev_terms (int, optional): highest degree of Chebyshev polynomials.
                            Must be even and at least 6. Defaults to 32.
            method (str, optional): method used to compute sigmoid function.
                Defaults to "reciprocal".

        Returns:
            NadaArray: The sigmoid evaluation.

        Raises:
            ValueError: Raised if method type is not supported.
        """
        if self.is_rational:

            def sigmoid(x):
                return x.sigmoid(chebyshev_terms=chebyshev_terms, method=method)

            return self.apply(sigmoid)

        dtype = get_dtype(self.inner)
        raise TypeError(
            f"Sigmoid is not compatible with {dtype},\
                only with Rational and SecretRational types."
        )

    def gelu(
        self, method: str = "tanh", tanh_method: str = "reciprocal"
    ) -> "NadaArray":
        r"""Computes the gelu function using the following definition

        .. math::
            gelu(x) = x/2 * (1 + tanh(\sqrt{2/\pi} * (x + 0.04471 * x^3)))

        Methods:
        If a valid method is given, this function will compute gelu
            using that method:

            "tanh" - computes gelu using the common approximation function

                Note: stable for x in [-18, 18]. Unstable otherwise.

            "motzkin" - computes gelu via approximation from the paper
                "BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers"
                on section 5.2 based on the Motzkin’s polynomial preprocessing technique.

                Note: stable for all input range as the approximation is truncated
                to relu function outside [-2.7, 2.7].

        Args:
            method (str, optional): method used to compute gelu function. Defaults to "tanh".
            tanh_method (str, optional): method used for tanh function. Defaults to "reciprocal".

        Returns:
            NadaArray: The gelu evaluation.

        Raises:
            ValueError: Raised if method type is not supported.
        """
        if self.is_rational:

            def gelu(x):
                return x.gelu(method=method, tanh_method=tanh_method)

            return self.apply(gelu)

        dtype = get_dtype(self.inner)
        raise TypeError(
            f"Gelu is not compatible with {dtype}, only with Rational and SecretRational types."
        )

    def silu(
        self,
        method_sigmoid: str = "reciprocal",
    ) -> "NadaArray":
        r"""Computes the gelu function using the following definition

        .. math::
            silu(x) = x * sigmoid(x)

        Sigmoid methods:
        If a valid method is given, this function will compute sigmoid
            using that method:

            "chebyshev" - computes tanh via Chebyshev approximation with
                truncation and uses the identity:

                .. math::
                    \sigma(x) = \frac{1}{2}tanh(\frac{x}{2}) + \frac{1}{2}

                Note: stable for all input range as the approximation is truncated
                        to 0/1 outside [-1, 1].

            "motzkin" - computes tanh via approximation from the paper
                "BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers"
                on section 5.3 based on the Motzkin’s polynomial preprocessing technique.
                It uses the identity:

                .. math::
                    \sigma(x) = \frac{1}{2}tanh(\frac{x}{2}) + \frac{1}{2}

                Note: stable for all input range as the approximation is truncated
                        to 0/1 outside [-1, 1].

            "reciprocal" - computes sigmoid using :math:`1 + e^{-x}` and computing
                the reciprocal

                Note: stable for x in [-500, 500]. Unstable otherwise.

        Args:
            method_sigmoid (str, optional): method used to compute sigmoid function.
                Defaults to "reciprocal".

        Returns:
            NadaArray: The sigmoid evaluation.

        Raises:
            ValueError: Raised if sigmoid method type is not supported.
        """
        if self.is_rational:

            def silu(x):
                return x.silu(method_sigmoid=method_sigmoid)

            return self.apply(silu)

        dtype = get_dtype(self.inner)
        raise TypeError(
            f"Silu is not compatible with {dtype}, only with Rational and SecretRational types."
        )


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


# Shuffle


def _butterfly_block(base: int, step: int) -> np.ndarray:
    """
    Generates a butterfly block of connections for a given base index and step.

    Parameters:
        base (int): The starting index for the butterfly block.
        step (int): The step size used to calculate the indices for connections.

    Returns:
        np.ndarray: A 2D array of connections where each row represents a pair of
            connected indices.
    """
    # Create a range of indices
    indices = np.arange(0, step, 2)
    # First half
    stage_i_1st_half = np.column_stack((base + indices, base + (indices + step)))
    # Second half
    stage_i_2nd_half = np.column_stack(
        (base + (indices + step + 1), base + (indices + 1))
    )
    # Concatenate the two halves
    return np.vstack((stage_i_1st_half, stage_i_2nd_half))


def _benes_network(n: int) -> np.ndarray:
    """
    Constructs the Benes network for a given number of inputs/outputs.

    Args:
        n (int): The number of inputs/outputs. Must be a power of 2.

    Returns:
        np.ndarray: A 3D array where each 2D array represents the connections for a stage in the
                network. Each row in the 2D array represents a pair of connected indices.
    """
    if (n & (n - 1)) != 0 or n <= 0:
        raise ValueError(
            f"Benes network generation error. You asked for a benes network on {n} elemenst.\
                             The number of inputs must be a power of 2 and greater than 0. "
        )

    stages = []
    log_n = int(np.log2(n))

    # Stage 0: Initial connections (adjacent pairs)
    indices = np.arange(0, n, 2)
    stage_0 = np.column_stack((indices, indices + 1))
    stages.append(stage_0)

    # Stages 1 to log_n:
    for stage_i in range(1, log_n):
        step = n // 2**stage_i  # index step between the first and second half of blocks
        nr_of_halfs = 2**stage_i
        stage_i_connections = []

        for idx in range(0, nr_of_halfs, 2):
            base = idx * step
            halves = _butterfly_block(base, step)
            stage_i_connections.append(halves)

        # Combine the connections for the current stage
        stage = np.vstack(stage_i_connections)
        stages.append(stage)

    # Reverse the stages for the second half
    stages += stages[1:][::-1]

    return np.array(stages)


def _rand_bool() -> SecretBoolean:
    """
    Generates a random boolean.
    """
    r = NadaArray.random((1,), SecretRational)[0]
    return r > rational(0)


_SwapTypes = Union[
    Rational,
    SecretRational,
    SecretInteger,
    PublicInteger,
    Integer,
    PublicUnsignedInteger,
    SecretUnsignedInteger,
]


def _swap_gate(a: _SwapTypes, b: _SwapTypes) -> Tuple[_SwapTypes, _SwapTypes]:
    """
    Conditionally swaps two secret-shared rational numbers using a random boolean value.
    """
    rbool = _rand_bool()
    # swap
    r1 = rbool.if_else(a, b)
    r2 = rbool.if_else(b, a)
    return r1, r2

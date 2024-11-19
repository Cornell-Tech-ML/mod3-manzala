from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

from numba import cuda
from numba import njit
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


@njit
def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
    ----
        index : index tuple of ints
        strides : tensor strides

    Returns:
    -------
        Position in storage

    """
    position = 0
    for i, j in zip(index, strides):
        position += i * j
    return position
    # raise NotImplementedError("Need to implement for Task 2.1")


@njit
def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
    ----
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    # TODO: Implement for Task 2.1.
    cur_ord = ordinal + 0
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur_ord % sh)
        cur_ord = cur_ord // sh
    # raise NotImplementedError("Need to implement for Task 2.1")


@njit
def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
    ----
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
    -------
        None

    """
    # TODO: Implement for Task 2.2.
    big_start_dim = len(big_shape) - len(shape)
    for i, position in enumerate(shape):
        if position <= 1 and i < len(out_index):
            out_index[i] = 0
        else:
            out_index[i] = big_index[i + big_start_dim]
    return

    # raise NotImplementedError("Need to implement for Task 2.2")


@njit
def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
    ----
        shape1 : first shape
        shape2 : second shape

    Returns:
    -------
        broadcasted shape

    Raises:
    ------
        IndexingError : if cannot broadcast

    """
    # TODO: Implement for Task 2.2.
    broadcast_shape = []
    if len(shape2) > len(shape1):
        shape1, shape2 = shape2, shape1
    shape2_new = [1] * (len(shape1) - len(shape2)) + list(shape2)
    for index, val in enumerate(shape2_new):
        if (shape1[index] == val) or (shape1[index] == 1) or (val == 1):
            broadcast_shape.append(max(val, shape1[index]))
        else:
            raise IndexingError("Cannot Broadcast")
    return tuple(broadcast_shape)
    # raise NotImplementedError("Need to implement for Task 2.2")


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Calculate the strides from a given shape.

    Args:
    ----
        shape : UserShape
            The shape for which to calculate the strides.

    Returns:
    -------
        UserStrides
            The calculated strides.

    """
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        """Initialize the TensorData object.

        Args:
        ----
            storage : Union[Sequence[float], Storage]
                The storage for the tensor.
            shape : UserShape
                The shape of the tensor.
            strides : Optional[UserStrides], optional
                The strides of the tensor. If not provided, they will be calculated from the shape.

        Raises:
        ------
            IndexingError : if the length of the strides does not match the shape.

        """
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:
        """Move the tensor storage to CUDA device if it's not already there.

        This method checks if the tensor storage is currently on a CUDA device. If not, it moves the storage to a CUDA device using Numba's CUDA support.
        """
        if not cuda.is_cuda_array(self._storage):
            self._storage = cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns
        -------
            bool : True if contiguous

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Broadcasts two shapes to a common shape according to broadcasting rules.

        This method takes two shapes and returns a new shape that is the result of broadcasting the two input shapes. The broadcasting rules are as follows:
        - If the two shapes do not have the same number of dimensions, the shape with the fewer dimensions is padded on the left with ones.
        - If the shape of the two arrays does not match in any dimension, the array with the size of one in that dimension is broadcasted.
        - If the size of the two arrays does not match in any dimension, the size of the output is the maximum size.

        Args:
        ----
            shape_a (UserShape): The first shape to broadcast.
            shape_b (UserShape): The second shape to broadcast.

        Returns:
        -------
            UserShape: The broadcasted shape.

        """
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Converts a given index to a position in the tensor storage.

        This method takes an index, which can be an integer or a tuple of integers, and converts it to a position in the tensor storage. It checks for errors such as mismatched dimensions, out-of-range values, and negative indexing, and raises an `IndexingError` if any of these conditions are met.

        Args:
        ----
            index (Union[int, UserIndex]): The index to convert. Can be an integer or a tuple of integers.

        Returns:
        -------
            int: The position in the tensor storage corresponding to the given index.

        Raises:
        ------
            TypeError: If the index type is not supported.
            IndexingError: If the index dimensions do not match the tensor shape, or if the index is out of range or negative.

        """
        if isinstance(index, int):
            aindex: Index = array([index])
        elif isinstance(index, tuple):
            aindex = array(index)
        else:
            raise TypeError(f"Unsupported index type: {type(index)}")

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Generates all possible indices for the tensor.

        This method iterates over the entire tensor, converting each ordinal position to a multidimensional index based on the tensor's shape. It yields each index as a tuple of integers.

        Yields
        ------
            UserIndex: A tuple of integers representing a valid index in the tensor.

        """
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Generates a random index within the bounds of the tensor's shape.

        This method generates a random integer for each dimension of the tensor, ensuring the generated index is within the valid range for each dimension. The generated index is returned as a tuple of integers.

        Returns
        -------
            UserIndex: A tuple of random integers representing a valid index within the tensor.

        """
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Retrieves the value at the specified index in the tensor.

        This method takes a tuple of integers as the index and returns the value at that index in the tensor's storage.

        Args:
        ----
            key (UserIndex): A tuple of integers representing the index in the tensor.

        Returns:
        -------
            float: The value at the specified index in the tensor.

        """
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Sets the value at the specified index in the tensor.

        This method updates the value in the tensor's storage at the index specified by `key` to `val`.

        Args:
        ----
            key (UserIndex): A tuple of integers representing the index in the tensor.
            val (float): The value to set at the specified index.

        Returns:
        -------
            None: This method does not return a value.

        """
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Returns a tuple containing the storage, shape, and strides of the tensor.

        This method provides a way to access the internal storage, shape, and strides of the tensor as a tuple. This can be useful for operations that require direct access to these attributes.

        Returns
        -------
            Tuple[Storage, Shape, Strides]: A tuple containing the storage, shape, and strides of the tensor.

        """
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
        ----
            order (list): a permutation of the dimensions

        Returns:
        -------
            New `TensorData` with the same storage and a new dimension order.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        return TensorData(
            self._storage,
            tuple([self.shape[int(i)] for i in order]),
            tuple([self.strides[int(i)] for i in order]),
        )

        # raise NotImplementedError("Need to implement for Task 2.1")

    def to_string(self) -> str:
        """Converts the tensor data to a string representation.

        This method iterates over all indices in the tensor, constructs a string representation of the tensor's structure, and inserts the values at each index. The string representation is formatted to visually represent the tensor's structure, with each dimension separated by a newline and indented with tabs.

        Returns
        -------
            str: A string representation of the tensor data.

        """
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s

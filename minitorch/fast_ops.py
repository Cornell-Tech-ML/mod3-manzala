from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# JIT-compiles optimized versions of tensor data functions


class FastOps(TensorOps):
    """Provides optimized operations on tensors using NUMBA for JIT compilation."""

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Applies a function element-wise to a tensor.

        Args:
        ----
            fn (Callable[[float], float]): The function to apply to each element.

        Returns:
        -------
            MapProto: A callable that applies the function to a tensor.

        """
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Applies a binary function element-wise to two tensors.

        Args:
        ----
            fn (Callable[[float, float], float]): The binary function to apply.

        Returns:
        -------
            Callable: A callable that applies the function to two tensors.

        """
        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduces a tensor along a specified dimension using a binary function.

        Args:
        ----
            fn (Callable[[float, float], float]): The binary reduction function.
            start (float): The initial value for the reduction.

        Returns:
        -------
            Callable: A callable that reduces a tensor along a dimension.

        """
        f = tensor_reduce(fn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start
            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Computes the batched tensor matrix multiplication.

        Args:
        ----
            a (Tensor): The first tensor.
            b (Tensor): The second tensor.

        Returns:
        -------
            Tensor: The result of the matrix multiplication.

        """
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA higher-order tensor map function.

    Applies a function element-wise to a tensor and stores the result in an output tensor.

    Args:
    ----
        fn (Callable[[float], float]): The function to apply.

    Returns:
    -------
        Callable: A callable that applies the map function to tensors.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        if list(in_shape) == list(out_shape) and list(in_strides) == list(out_strides):
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        else:
            for i in prange(len(out)):
                out_index = np.zeros(MAX_DIMS, np.int32)
                in_index = np.zeros(MAX_DIMS, np.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                data = in_storage[index_to_position(in_index, in_strides)]
                out[index_to_position(out_index, out_strides)] = fn(data)

    return njit(parallel=True)(_map)


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function.

    Applies a binary function element-wise to two tensors and stores the result in an output tensor.

    Args:
    ----
        fn (Callable[[float, float], float]): The binary function to apply.

    Returns:
    -------
        Callable: A callable that applies the zip function to tensors.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        if list(a_strides) == list(b_strides) == list(out_strides) and list(
            a_shape
        ) == list(b_shape) == list(out_shape):
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            for i in prange(len(out)):
                a_index = np.zeros(MAX_DIMS, np.int32)
                b_index = np.zeros(MAX_DIMS, np.int32)
                o_index = np.zeros(MAX_DIMS, np.int32)
                to_index(i, out_shape, o_index)
                broadcast_index(o_index, out_shape, a_shape, a_index)
                broadcast_index(o_index, out_shape, b_shape, b_index)
                a_data = a_storage[index_to_position(a_index, a_strides)]
                b_data = b_storage[index_to_position(b_index, b_strides)]
                map_data = fn(a_data, b_data)
                out[index_to_position(o_index, out_strides)] = map_data

    return njit(parallel=True)(_zip)


@njit
def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function.

    Reduces a tensor along a specified dimension using a binary function.

    Args:
    ----
        fn (Callable[[float, float], float]): The binary reduction function.

    Returns:
    -------
        Callable: A callable that reduces a tensor along a dimension.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        reduce_size = a_shape[reduce_dim]
        for i in prange(len(out)):
            out_index = np.zeros(MAX_DIMS, np.int32)
            to_index(i, out_shape, out_index)
            o_pos = index_to_position(out_index, out_strides)
            acc = out[o_pos]
            for j in range(reduce_size):
                out_index[reduce_dim] = j
                a_pos = index_to_position(out_index, a_strides)
                acc = fn(acc, a_storage[a_pos])
            out[o_pos] = acc

    return njit(parallel=True)(_reduce)


@njit(parallel=True)
def tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Multiplies two matrices or tensors along their last two dimensions.

    Args:
    ----
        out (Storage): Storage for the output tensor.
        out_shape (Shape): Shape of the output tensor.
        out_strides (Strides): Strides of the output tensor.
        a_storage (Storage): Storage for tensor A.
        a_shape (Shape): Shape of tensor A.
        a_strides (Strides): Strides of tensor A.
        b_storage (Storage): Storage for tensor B.
        b_shape (Shape): Shape of tensor B.
        b_strides (Strides): Strides of tensor B.

    """
    assert a_shape[-1] == b_shape[-2]
    stride_batch_a = a_strides[0] if a_shape[0] > 1 else 0
    stride_batch_b = b_strides[0] if b_shape[0] > 1 else 0

    stride_row_a = a_strides[2]
    stride_col_b = b_strides[1]
    sum_limit = a_shape[-1]

    for idx in prange(out_shape[0]):
        for row in range(out_shape[1]):
            for col in range(out_shape[2]):
                sum_temp = 0.0
                a_index = idx * stride_batch_a + row * a_strides[1]
                b_index = idx * stride_batch_b + col * b_strides[2]

                for _ in range(sum_limit):
                    sum_temp += a_storage[a_index] * b_storage[b_index]
                    a_index += stride_row_a
                    b_index += stride_col_b

                result_index = (
                    idx * out_strides[0] + row * out_strides[1] + col * out_strides[2]
                )
                out[result_index] = sum_temp

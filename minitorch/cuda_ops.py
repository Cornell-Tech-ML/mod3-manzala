# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    """
    Wraps a function with `numba.cuda.jit` for device-level compilation.

    Args:
        fn (Callable): The function to compile.

    Returns:
        Callable: The compiled device function.
    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    """
    Wraps a function with `numba.cuda.jit` for kernel-level compilation.

    Args:
        fn (Callable): The function to compile.

    Returns:
        Callable: The compiled kernel function.
    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


def tensor_map(fn):
    """
    CUDA higher-order tensor map function.

    Maps a function to each element of an input tensor and stores the result
    in an output tensor.

    Args:
        fn (Callable[[float], float]): The function to map.

    Returns:
        Callable: The CUDA kernel for mapping.
    """

    def _map(out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides):
        i = numba.cuda.blockIdx.x * THREADS_PER_BLOCK + numba.cuda.threadIdx.x
        if i < out_size:
            out_index = numba.cuda.local.array(MAX_DIMS, numba.int32)
            in_index = numba.cuda.local.array(MAX_DIMS, numba.int32)
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_position = index_to_position(in_index, in_strides)
            out_position = index_to_position(out_index, out_strides)
            out[out_position] = fn(in_storage[in_position])

    return cuda.jit()(_map)


def tensor_zip(fn):
    """
    CUDA higher-order tensor zip function.

    Applies a binary function to pairs of elements from two input tensors
    and stores the result in an output tensor.

    Args:
        fn (Callable[[float, float], float]): The binary function to apply.

    Returns:
        Callable: The CUDA kernel for zipping.
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        i = numba.cuda.blockIdx.x * THREADS_PER_BLOCK + numba.cuda.threadIdx.x
        if i < out_size:
            out_index = numba.cuda.local.array(MAX_DIMS, numba.int32)
            a_index = numba.cuda.local.array(MAX_DIMS, numba.int32)
            b_index = numba.cuda.local.array(MAX_DIMS, numba.int32)
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_position = index_to_position(a_index, a_strides)
            b_position = index_to_position(b_index, b_strides)
            out_position = index_to_position(out_index, out_strides)
            out[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return cuda.jit()(_zip)


def tensor_reduce(fn):
    """
    CUDA higher-order tensor reduce function.

    Reduces a tensor along a specified dimension using a binary reduction
    function.

    Args:
        fn (Callable[[float, float], float]): The binary reduction function.

    Returns:
        Callable: The CUDA kernel for reduction.
    """

    def _reduce(
        out,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        reduce_dim,
        reduce_value,
    ):
        BLOCK_DIM = 1024
        local_idx = numba.cuda.threadIdx.x
        block_idx = numba.cuda.blockIdx.x
        shared_block = numba.cuda.shared.array(BLOCK_DIM, numba.float64)
        offset = 1

        out_index = numba.cuda.local.array(MAX_DIMS, numba.int32)
        to_index(block_idx, out_shape, out_index)
        out_position = index_to_position(out_index, out_strides)

        if local_idx < a_shape[reduce_dim]:
            out_index[reduce_dim] = local_idx
            shared_block[local_idx] = a_storage[index_to_position(out_index, a_strides)]
        else:
            shared_block[local_idx] = reduce_value

        while offset < BLOCK_DIM:
            numba.cuda.syncthreads()
            if local_idx % (offset * 2) == 0:
                shared_block[local_idx] = fn(
                    shared_block[local_idx], shared_block[local_idx + offset]
                )
            offset *= 2

        numba.cuda.syncthreads()
        if local_idx == 0:
            out[out_position] = shared_block[local_idx]

    return cuda.jit()(_reduce)


def map(fn):
    """
    Creates a higher-order map function for tensors.

    Args:
        fn (Callable[[float], float]): The unary function to map.

    Returns:
        Callable: A function that applies the map to tensors.
    """
    f = tensor_map(cuda.jit(device=True)(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())
        return out

    return ret


def zip(fn):
    """
    Creates a higher-order zip function for tensors.

    Args:
        fn (Callable[[float, float], float]): The binary function to zip.

    Returns:
        Callable: A function that applies the zip to tensors.
    """
    f = tensor_zip(cuda.jit(device=True)(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
        f[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )
        return out

    return ret


def reduce(fn, start=0.0):
    """
    Creates a higher-order reduce function for tensors.

    Args:
        fn (Callable[[float, float], float]): The binary reduction function.
        start (float): The initial value for reduction.

    Returns:
        Callable: A function that reduces tensors.
    """
    f = tensor_reduce(cuda.jit(device=True)(fn))

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
        out_a = a.zeros(tuple(out_shape))

        threadsperblock = 1024
        blockspergrid = out_a.size
        f[blockspergrid, threadsperblock](
            *out_a.tuple(), out_a.size, *a.tuple(), dim, start
        )

        return out_a

    return ret


def _sum_practice(out, a, size):
    """
    Practice implementation for sum reduction using shared memory.

    Args:
        out (array): Output tensor for partial sums.
        a (array): Input tensor to reduce.
        size (int): Size of the input tensor.
    """
    BLOCK_DIM = 32
    shared = numba.cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * THREADS_PER_BLOCK + cuda.threadIdx.x
    local_idx = cuda.threadIdx.x

    if i < size:
        shared[local_idx] = a[i]
    else:
        shared[local_idx] = 0

    cuda.syncthreads()

    offset = 1
    while offset < BLOCK_DIM:
        if local_idx % (2 * offset) == 0 and local_idx + offset < BLOCK_DIM:
            shared[local_idx] += shared[local_idx + offset]
        offset *= 2
        cuda.syncthreads()

    if local_idx == 0:
        out[cuda.blockIdx.x] = shared[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a):
    """
    Wrapper function for practicing sum reduction.

    Args:
        a (Tensor): Input tensor to reduce.

    Returns:
        Tensor: Partially reduced tensor.
    """
    size = a.shape[0]
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    out = TensorData([0.0] * blockspergrid, (blockspergrid,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out._tensor._storage, a._tensor._storage, size
    )
    return out


@cuda.jit()
def tensor_matrix_multiply(
    out, out_shape, out_strides, out_size, a_storage, a_shape, a_strides, b_storage, b_shape, b_strides
):
    """
    CUDA implementation of tensor matrix multiplication.

    Args:
        out (array): Output tensor storage.
        out_shape (array): Shape of the output tensor.
        out_strides (array): Strides of the output tensor.
        out_size (int): Size of the output tensor.
        a_storage (array): Storage for tensor A.
        a_shape (array): Shape of tensor A.
        a_strides (array): Strides of tensor A.
        b_storage (array): Storage for tensor B.
        b_shape (array): Shape of tensor B.
        b_strides (array): Strides of tensor B.
    """
    BLOCK_DIM = 32
    shared_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shared_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    x = cuda.threadIdx.x + cuda.blockIdx.x * BLOCK_DIM
    y = cuda.threadIdx.y + cuda.blockIdx.y * BLOCK_DIM

    for block_k in range((a_shape[-1] + BLOCK_DIM - 1) // BLOCK_DIM):
        shared_a[cuda.threadIdx.y, cuda.threadIdx.x] = 0
        shared_b[cuda.threadIdx.y, cuda.threadIdx.x] = 0

        # Load shared memory
        if x < a_shape[-2] and block_k * BLOCK_DIM + cuda.threadIdx.x < a_shape[-1]:
            shared_a[cuda.threadIdx.y, cuda.threadIdx.x] = a_storage[
                index_to_position((x, block_k * BLOCK_DIM + cuda.threadIdx.x), a_strides)
            ]
        if y < b_shape[-1] and block_k * BLOCK_DIM + cuda.threadIdx.y < b_shape[-2]:
            shared_b[cuda.threadIdx.y, cuda.threadIdx.x] = b_storage[
                index_to_position((block_k * BLOCK_DIM + cuda.threadIdx.y, y), b_strides)
            ]
        cuda.syncthreads()

        for k in range(BLOCK_DIM):
            out[index_to_position((x, y), out_strides)] += shared_a[cuda.threadIdx.y, k] * shared_b[k, cuda.threadIdx.x]
        cuda.syncthreads()


class CudaOps(TensorOps):
    """
    Class for implementing CUDA operations on tensors.
    """

    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """
        Applies a unary function to each element of a tensor.

        Args:
            fn (Callable[[float], float]): The function to apply.

        Returns:
            Callable: A function that applies the map operation.
        """
        return map(device_jit(fn))

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """
        Applies a binary function to pairs of elements from two tensors.

        Args:
            fn (Callable[[float, float], float]): The binary function to apply.

        Returns:
            Callable: A function that applies the zip operation.
        """
        return zip(device_jit(fn))

    @staticmethod
    def reduce(fn: Callable[[float, float], float], start=0.0) -> Callable[[Tensor, int], Tensor]:
        """
        Reduces a tensor along a specified dimension.

        Args:
            fn (Callable[[float, float], float]): The reduction function to apply.
            start (float): The initial value for the reduction.

        Returns:
            Callable: A function that performs the reduction.
        """
        return reduce(device_jit(fn), start)

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Multiplies two matrices or tensors.

        Args:
            a (Tensor): The first tensor.
            b (Tensor): The second tensor.

        Returns:
            Tensor: The resulting tensor after multiplication.
        """
        blocks = (a.shape[0], b.shape[1], 1)
        threads = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)
        out = Tensor.zeros((a.shape[0], b.shape[1]))
        tensor_matrix_multiply[blocks, threads](
            out._storage,
            out.shape,
            out.strides,
            out.size,
            a._storage,
            a.shape,
            a.strides,
            b._storage,
            b.shape,
            b.strides,
        )
        return out

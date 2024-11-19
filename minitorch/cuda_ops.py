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


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a function for CUDA device execution.

    Args:
    ----
        fn: The function to compile.
        kwargs: Additional keyword arguments for the JIT compiler.

    Returns:
    -------
        The JIT compiled function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Callable, **kwargs: Any) -> FakeCUDAKernel:
    """JIT compile a function for CUDA execution.

    Args:
    ----
        fn: The function to compile.
        kwargs: Additional keyword arguments for the JIT compiler.

    Returns:
    -------
        A CUDA kernel.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    """CUDA operations for tensor computations."""

    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Create a CUDA map operation for a given function.

        Args:
        ----
            fn: A function that maps a float to a float.

        Returns:
        -------
            A callable that applies the map operation to tensors.

        """
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            """Apply the map operation to a tensor.

            Args:
            ----
                a: The input tensor.
                out: Optional output tensor to store the result.

            Returns:
            -------
                The tensor with the map operation applied.

            """
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Create a CUDA zip operation for a given function.

        Args:
        ----
            fn: A function that maps two floats to a float.

        Returns:
        -------
            A callable that applies the zip operation to tensors.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            """Apply the zip operation to two tensors.

            Args:
            ----
                a: The first input tensor.
                b: The second input tensor.

            Returns:
            -------
                The tensor with the zip operation applied.

            """
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Create a CUDA reduce operation for a given function.

        Args:
        ----
            fn: A function that reduces two floats to a float.
            start: The initial value for the reduction.

        Returns:
        -------
            A callable that applies the reduce operation to tensors.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            """Apply the reduce operation to a tensor along a specified dimension.

            Args:
            ----
                a: The input tensor.
                dim: The dimension along which to reduce.

            Returns:
            -------
                The reduced tensor.

            """
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Perform matrix multiplication on two tensors using CUDA.

        Args:
        ----
            a: The first tensor.
            b: The second tensor.

        Returns:
        -------
            The result of the matrix multiplication.

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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, int, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function.

    Maps a function to each element of an input tensor and stores the result
    in an output tensor.

    Args:
    ----
        fn (Callable[[float], float]): The function to map.

    Returns:
    -------
        Callable: The CUDA kernel for mapping.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
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


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, int, Storage, Shape, Strides, Storage, Shape, Strides],
    None,
]:
    """CUDA higher-order tensor zip function.

    Applies a binary function to pairs of elements from two input tensors
    and stores the result in an output tensor.

    Args:
    ----
        fn (Callable[[float, float], float]): The binary function to apply.

    Returns:
    -------
        Callable: The CUDA kernel for zipping.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
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


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, int, Storage, Shape, Strides, int, float], None
]:
    """CUDA higher-order tensor reduce function.

    Reduces a tensor along a specified dimension using a binary reduction
    function.

    Args:
    ----
        fn (Callable[[float, float], float]): The binary reduction function.

    Returns:
    -------
        Callable: The CUDA kernel for reduction.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
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


def map(fn: Callable[[float], float]) -> Callable[[Tensor, Optional[Tensor]], Tensor]:
    """Creates a higher-order map function for tensors.

    Args:
    ----
        fn (Callable[[float], float]): The unary function to map.

    Returns:
    -------
        Callable[[Tensor, Optional[Tensor]], Tensor]: A function that applies the map to tensors.

    """
    f = tensor_map(cuda.jit(device=True)(fn))

    def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
        if out is None:
            out = a.zeros(a.shape)
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())
        return out

    return ret


def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
    """Creates a higher-order zip function for tensors.

    Args:
    ----
        fn (Callable[[float, float], float]): The binary function to zip.

    Returns:
    -------
        Callable[[Tensor, Tensor], Tensor]: A function that applies the zip to tensors.

    """
    f = tensor_zip(cuda.jit(device=True)(fn))

    def ret(a: Tensor, b: Tensor) -> Tensor:
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
        f[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )
        return out

    return ret


def reduce(
    fn: Callable[[float, float], float], start: float = 0.0
) -> Callable[[Tensor, int], Tensor]:
    """Creates a higher-order reduce function for tensors.

    Args:
    ----
        fn (Callable[[float, float], float]): The binary reduction function.
        start (float): The initial value for reduction.

    Returns:
    -------
        Callable[[Tensor, int], Tensor]: A function that reduces tensors.

    """
    f = tensor_reduce(cuda.jit(device=True)(fn))

    def ret(a: Tensor, dim: int) -> Tensor:
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


def _sum_practice(out: Any, a: Any, size: Any) -> None:
    """Practice implementation for sum reduction using shared memory.

    Args:
    ----
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


def sum_practice(a: Any) -> Any:
    """Wrapper function for practicing sum reduction.

    Args:
    ----
        a (Tensor): Input tensor to reduce.

    Returns:
    -------
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


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

      * All data must be first moved to shared memory.
      * Only read each cell in `a` and `b` once.
      * Only write to global memory once per kernel.

    Compute ::

    for i:
        for j:
             for k:
                 out[i, j] += a[i, k] * b[k, j]

    Args:
    ----
        out (array): storage for `out` tensor.
        a (array): storage for `a` tensor.
        b (array): storage for `a` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    shared_a = numba.cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shared_b = numba.cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    y = numba.cuda.threadIdx.y
    x = numba.cuda.threadIdx.x
    if x < size and y < size:
        shared_a[y, x] = a[y * size + x]
        shared_b[y, x] = b[y * size + x]
    else:
        shared_a[y, x] = 0
        shared_b[y, x] = 0
    numba.cuda.syncthreads()

    if y < size and x < size:
        temp = 0
        for val in range(size):
            temp += shared_a[y, val] * shared_b[val, x]
        out[y * size + x] = temp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Performs matrix multiplication between two tensors `a` and `b` using CUDA.

    This function computes the matrix product of two tensors `a` and `b` and returns the result as a new `TensorData` object. It utilizes CUDA acceleration for the computation.

    Args:
    ----
        a (Tensor): The first tensor for matrix multiplication.
        b (Tensor): The second tensor for matrix multiplication.

    Returns:
    -------
        TensorData: The result of the matrix multiplication as a `TensorData` object.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


@cuda.jit()
def tensor_matrix_multiply(
    out: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    out_shape: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    out_strides: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    out_size: int,
    a_storage: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    a_shape: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    a_strides: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    b_storage: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    b_shape: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    b_strides: numba.cuda.cudadrv.devicearray.DeviceNDArray,
) -> None:
    """CUDA implementation of tensor matrix multiplication.

    Args:
    ----
        out (DeviceNDArray): Output tensor storage.
        out_shape (DeviceNDArray): Shape of the output tensor.
        out_strides (DeviceNDArray): Strides of the output tensor.
        out_size (int): Size of the output tensor.
        a_storage (DeviceNDArray): Storage for tensor A.
        a_shape (DeviceNDArray): Shape of tensor A.
        a_strides (DeviceNDArray): Strides of tensor A.
        b_storage (DeviceNDArray): Storage for tensor B.
        b_shape (DeviceNDArray): Shape of tensor B.
        b_strides (DeviceNDArray): Strides of tensor B.

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
                index_to_position(
                    (x, block_k * BLOCK_DIM + cuda.threadIdx.x), a_strides
                )
            ]
        if y < b_shape[-1] and block_k * BLOCK_DIM + cuda.threadIdx.y < b_shape[-2]:
            shared_b[cuda.threadIdx.y, cuda.threadIdx.x] = b_storage[
                index_to_position(
                    (block_k * BLOCK_DIM + cuda.threadIdx.y, y), b_strides
                )
            ]
        cuda.syncthreads()

        for k in range(BLOCK_DIM):
            out[index_to_position((x, y), out_strides)] += (
                shared_a[cuda.threadIdx.y, k] * shared_b[k, cuda.threadIdx.x]
            )
        cuda.syncthreads()

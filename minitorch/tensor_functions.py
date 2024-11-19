"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Tuple

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> Tuple[Any]:
    """Convert a value to a tuple if it is not already a tuple.

    Args:
    ----
        x: The value to be converted.

    Returns:
    -------
        A tuple containing the value.

    """
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        """Compute the backward pass for the function.

        Args:
        ----
            ctx: The context containing saved values for backward computation.
            grad_out: The gradient of the output.

        Returns:
        -------
            A tuple of gradients with respect to the inputs.

        """
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        """Compute the forward pass for the function.

        Args:
        ----
            ctx: The context to save values for backward computation.
            *inps: The input tensors.

        Returns:
        -------
            The result of the forward computation as a Tensor.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Apply the function to the given input tensors and track history.

        Args:
        ----
            *vals: The input tensors.

        Returns:
        -------
            A new tensor with the result of the function and history for backpropagation.

        """
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the negative of the input tensor.

        Args:
        ----
            ctx: The context for saving values.
            t1: The input tensor.

        Returns:
        -------
            The negative of the input tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the negative function.

        Args:
        ----
            ctx: The context with saved values.
            grad_output: The gradient of the output.

        Returns:
        -------
            The gradient of the input.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the inverse of the input tensor.

        Args:
        ----
            ctx: The context for saving values.
            t1: The input tensor.

        Returns:
        -------
            The inverse of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the inverse function.

        Args:
        ----
            ctx: The context with saved values.
            grad_output: The gradient of the output.

        Returns:
        -------
            The gradient of the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the element-wise addition of two tensors.

        Args:
        ----
            ctx: The context for saving values.
            t1: The first input tensor.
            t2: The second input tensor.

        Returns:
        -------
            The result of adding the two tensors.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the addition function.

        Args:
        ----
            ctx: The context with saved values.
            grad_output: The gradient of the output.

        Returns:
        -------
            A tuple of gradients with respect to the inputs.

        """
        return grad_output, grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compute the element-wise multiplication of two tensors.

        Args:
        ----
            ctx: The context for saving values.
            a: The first input tensor.
            b: The second input tensor.

        Returns:
        -------
            The result of multiplying the two tensors.

        """
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the multiplication function.

        Args:
        ----
            ctx: The context with saved values.
            grad_output: The gradient of the output.

        Returns:
        -------
            A tuple of gradients with respect to the inputs.

        """
        (a, b) = ctx.saved_values
        return grad_output.f.mul_zip(b, grad_output), grad_output.f.mul_zip(
            a, grad_output
        )
        # raise NotImplementedError("Need to implement for Task 2.4")


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the sigmoid of the input tensor.

        Args:
        ----
            ctx: The context for saving values.
            t1: The input tensor.

        Returns:
        -------
            The sigmoid of the input tensor.

        """
        res = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(res)
        return res
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the sigmoid function.

        Args:
        ----
            ctx: The context with saved values.
            grad_output: The gradient of the output.

        Returns:
        -------
            The gradient of the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.mul_zip(
            grad_output,
            grad_output.f.mul_zip(
                grad_output.f.add_zip(
                    minitorch.Tensor.make([1.0], (1,), backend=grad_output.backend),
                    grad_output.f.neg_map(t1),
                ),
                t1,
            ),
        )

        # raise NotImplementedError("Need to implement for Task 2.4")


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the ReLU of the input tensor.

        Args:
        ----
            ctx: The context for saving values.
            t1: The input tensor.

        Returns:
        -------
            The ReLU of the input tensor.

        """
        res = t1.f.relu_map(t1)
        ctx.save_for_backward(res)
        return res
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the ReLU function.

        Args:
        ----
            ctx: The context with saved values.
            grad_output: The gradient of the output.

        Returns:
        -------
            The gradient of the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)
        # raise NotImplementedError("Need to implement for Task 2.4")


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the natural logarithm of the input tensor.

        Args:
        ----
            ctx: The context for saving values.
            t1: The input tensor.

        Returns:
        -------
            The natural logarithm of the input tensor.

        """
        res = t1.f.log_map(t1)
        ctx.save_for_backward(t1)
        return res
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the logarithm function.

        Args:
        ----
            ctx: The context with saved values.
            grad_output: The gradient of the output.

        Returns:
        -------
            The gradient of the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)
        # raise NotImplementedError("Need to implement for Task 2.4")


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the exponential of the input tensor.

        Args:
        ----
            ctx: The context for saving values.
            t1: The input tensor.

        Returns:
        -------
            The exponential of the input tensor.

        """
        res = t1.f.exp_map(t1)
        ctx.save_for_backward(t1)
        return res
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the exponential function.

        Args:
        ----
            ctx: The context with saved values.
            grad_output: The gradient of the output.

        Returns:
        -------
            The gradient of the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.mul_zip(t1, grad_output.f.exp_map(t1))
        # raise NotImplementedError("Need to implement for Task 2.4")


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Compute the sum of the input tensor along a specified dimension.

        Args:
        ----
            ctx: The context for saving values.
            a: The input tensor.
            dim: The dimension along which to sum.

        Returns:
        -------
            The sum of the input tensor along the specified dimension.

        """
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the sum function.

        Args:
        ----
            ctx: The context with saved values.
            grad_output: The gradient of the output.

        Returns:
        -------
            A tuple of the gradient of the input and a float.

        """
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all elements are true along a specified dimension.

        Args:
        ----
            ctx: The context for saving values.
            a: The input tensor.
            dim: The dimension along which to check.

        Returns:
        -------
            A tensor with 1 if all elements are true, otherwise 0.

        """
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compute the element-wise less-than comparison of two tensors.

        Args:
        ----
            ctx: The context for saving values.
            a: The first input tensor.
            b: The second input tensor.

        Returns:
        -------
            A tensor with 1 where a < b, otherwise 0.

        """
        res = a.f.lt_zip(a, b)
        ctx.save_for_backward(a, b)
        return res
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the less-than function.

        Args:
        ----
            ctx: The context with saved values.
            grad_output: The gradient of the output.

        Returns:
        -------
            A tuple of zero tensors with the same shape as the input.

        """
        (a, b) = ctx.saved_values
        return zeros(grad_output.shape), zeros(grad_output.shape)
        # raise NotImplementedError("Need to implement for Task 2.4")


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compute the element-wise equality comparison of two tensors.

        Args:
        ----
            ctx: The context for saving values.
            a: The first input tensor.
            b: The second input tensor.

        Returns:
        -------
            A tensor with 1 where a == b, otherwise 0.

        """
        res = a.f.eq_zip(a, b)
        ctx.save_for_backward(a, b)
        return res
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the equality function.

        Args:
        ----
            ctx: The context with saved values.
            grad_output: The gradient of the output.

        Returns:
        -------
            A tuple of zero tensors with the same shape as the input.

        """
        (a, b) = ctx.saved_values
        return zeros(grad_output.shape), zeros(grad_output.shape)
        # raise NotImplementedError("Need to implement for Task 2.4")


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compute the element-wise is-close comparison of two tensors.

        Args:
        ----
            ctx: The context for saving values.
            a: The first input tensor.
            b: The second input tensor.

        Returns:
        -------
            A tensor with 1 where a is close to b, otherwise 0.

        """
        res = a.f.is_close_zip(a, b)
        ctx.save_for_backward(a, b)
        return res
        # raise NotImplementedError("Need to implement for Task 2.3")


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Permute the dimensions of the input tensor according to the given order.

        Args:
        ----
            ctx: The context for saving values.
            a: The input tensor.
            order: The order to permute the dimensions.

        Returns:
        -------
            A new tensor with permuted dimensions.

        """
        rev_order = np.array([0 for _ in range(len(order._tensor._storage))])

        for i, j in enumerate(order._tensor._storage):
            rev_order[int(j)] = i

        ctx.save_for_backward(rev_order)

        return minitorch.Tensor(
            a._tensor.permute(*order._tensor._storage), backend=a.backend
        )
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the permute function.

        Args:
        ----
            ctx: The context with saved values.
            grad_output: The gradient of the output.

        Returns:
        -------
            A tuple of the permuted gradient and a float.

        """
        (rev_order,) = ctx.saved_values

        return (
            minitorch.Tensor(
                grad_output._tensor.permute(*rev_order), backend=grad_output.backend
            ),
            0.0,
        )
        # raise NotImplementedError("Need to implement for Task 2.4")


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Reshape the input tensor to the specified shape.

        Args:
        ----
            ctx: The context for saving values.
            a: The input tensor.
            shape: The new shape for the tensor.

        Returns:
        -------
            A new tensor with the specified shape.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the view function.

        Args:
        ----
            ctx: The context with saved values.
            grad_output: The gradient of the output.

        Returns:
        -------
            A tuple of the reshaped gradient and a float.

        """
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Return a copy of the input tensor.

        Args:
        ----
            ctx: The context for saving values.
            a: The input tensor.

        Returns:
        -------
            A copy of the input tensor.

        """
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the copy function.

        Args:
        ----
            ctx: The context with saved values.
            grad_output: The gradient of the output.

        Returns:
        -------
            The gradient of the input.

        """
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the matrix multiplication of two tensors.

        Args:
        ----
            ctx: The context for saving values.
            t1: The first input tensor.
            t2: The second input tensor.

        Returns:
        -------
            The result of matrix multiplication.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the matrix multiplication function.

        Args:
        ----
            ctx: The context with saved values.
            grad_output: The gradient of the output.

        Returns:
        -------
            A tuple of gradients with respect to the inputs.

        """
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of the specified shape.

    Args:
    ----
        shape: The shape of the tensor.
        backend: The tensor backend to use.

    Returns:
    -------
        A new tensor filled with zeros.

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of the specified shape.

    Args:
    ----
        shape: The shape of the tensor.
        backend: The tensor backend to use.
        requires_grad: Whether to enable autodifferentiation.

    Returns:
    -------
        A new tensor filled with random values.

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with specified data and shape.

    Args:
    ----
        ls: The data for the tensor.
        shape: The shape of the tensor.
        backend: The tensor backend to use.
        requires_grad: Whether to enable autodifferentiation.

    Returns:
    -------
        A new tensor with the specified data and shape.

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape inferred from the input list.

    Args:
    ----
        ls: The data for the tensor.
        backend: The tensor backend to use.
        requires_grad: Whether to enable autodifferentiation.

    Returns:
    -------
        A new tensor with the specified data and inferred shape.

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the central difference approximation of the gradient.

    Args:
    ----
        f: The function to differentiate.
        *vals: The input tensors.
        arg: The index of the argument to differentiate with respect to.
        epsilon: The small perturbation for finite difference.
        ind: The index at which to compute the gradient.

    Returns:
    -------
        The central difference approximation of the gradient.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Perform a gradient check for the given function and input tensors.

    Args:
    ----
        f: The function to check.
        *vals: The input tensors.

    Raises:
    ------
        AssertionError: If the computed gradient does not match the central difference approximation.

    """
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )

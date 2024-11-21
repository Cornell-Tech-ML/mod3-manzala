from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple."""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Perform the backward pass for the scalar function."""
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Perform the forward pass for the scalar function."""
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to a set of input values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Calculate the forward propagation."""
        return float(a + b)  # Ensure the result is a float

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Calculate the backward propagation."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Calculate the forward propagation."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculate the backward propagation."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Calculate the forward propagation."""
        ctx.save_for_backward(a, b)
        return float(a * b)  # Ensure the result is a float

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Calculate the backward propagation."""
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Calculate the forward propagation."""
        ctx.save_for_backward(a)
        return 1 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculate the backward propagation."""
        (a,) = ctx.saved_values
        return -d_output / (a**2)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Calculate the forward propagation."""
        return float(-a)  # Ensure the result is a float

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculate the backward propagation."""
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + exp(-x))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Calculate the forward propagation."""
        result = operators.sigmoid(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculate the backward propagation."""
        (result,) = ctx.saved_values
        return d_output * result * (1 - result)


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Calculate the forward propagation."""
        ctx.save_for_backward(a)
        # Ensure that the return value is always a float
        return float(max(0, a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculate the backward propagation."""
        (a,) = ctx.saved_values
        return d_output if a > 0 else 0.0


class Exp(ScalarFunction):
    """Exponential function $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Calculate the forward propagation."""
        result = operators.exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculate the backward propagation."""
        (result,) = ctx.saved_values
        return d_output * result


class LT(ScalarFunction):
    """Less-than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Calculate the forward propagation."""
        return float(1.0 if a < b else 0.0)  # Ensure the result is a float

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Calculate the backward propagation."""
        return 0.0, 0.0  # No gradient for comparison operators


class EQ(ScalarFunction):
    """Equality function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Calculate the forward propagation."""
        return float(1.0 if a == b else 0.0)  # Ensure the result is a float

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Calculate the backward propagation."""
        return 0.0, 0.0  # No gradient for comparison operators

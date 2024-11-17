from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b: ScalarLike) -> Scalar:
        return Add.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        return Add.apply(self, Neg.apply(b))

    def __neg__(self) -> Scalar:
        return Neg.apply(self)

    def __lt__(self, b: ScalarLike) -> Scalar:
        return LT.apply(self, b)

    def __eq__(self, b: ScalarLike) -> Scalar:
        return EQ.apply(self, b)

    def log(self) -> Scalar:
        """Compute the logarithm of the scalar."""
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Compute the exponential of the scalar."""
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Compute the sigmoid of the scalar."""
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Compute the ReLU of the scalar."""
        return ReLU.apply(self)

    def __bool__(self) -> bool:
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Return True if the scalar is constant (i.e., has no history)."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Return the parent variables that were used to compute this scalar."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to propagate gradients.

        Args:
        ----
            d_output: The derivative of the final output with respect to this variable.

        Returns:
        -------
            A list of tuples, where each tuple contains a parent variable and the
            derivative of the final output with respect to that parent.

        """
        # Ensure this variable has a history
        h = self.history
        assert h is not None, "No history found for this variable."
        assert h.last_fn is not None, "No last function in the history."
        assert h.ctx is not None, "No context was saved during the forward pass."

        fn = h.last_fn
        ctx = h.ctx
        inputs = h.inputs

        # Compute the local gradients
        local_derivs = fn.backward(ctx, d_output)

        # Ensure the result is iterable, even if it's a single value
        if isinstance(local_derivs, float):
            local_derivs = (local_derivs,)

        return zip(inputs, local_derivs)

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works correctly on a Python function.
    Asserts False if the derivative is incorrect.

    Parameters
    ----------
    f : Callable
        A function that takes scalar inputs and returns a scalar output.
    *scalars : Scalar
        One or more scalar values to check the derivative for.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )

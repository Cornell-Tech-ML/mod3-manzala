from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_plus = list(vals)
    vals_minus = list(vals)

    # Increment and decrement the specified argument
    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon

    # Compute the central difference approximation
    diff = (f(*vals_plus) - f(*vals_minus)) / (2 * epsilon)

    return diff


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative of the variable.

        Args:
        ----
            x: The derivative value to accumulate.

        """

    @property
    def unique_id(self) -> int:
        """Return the unique identifier of the variable."""
        return 1  # Replace with actual logic

    def is_leaf(self) -> bool:
        """Return whether the variable is a leaf node."""
        return True  # Replace with actual logic

    def is_constant(self) -> bool:
        """Return whether the variable is constant."""
        return False  # Replace with actual logic

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return the parent variables of this variable in the computation graph."""
        return []  # Replace with actual logic

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to propagate gradients for the given output.

        Args:
        ----
            d_output: The derivative of the output with respect to this variable.

        Returns:
        -------
            An iterable of tuples, where each tuple contains a parent variable
            and the local derivative with respect to that parent.

        """
        return []  # Replace with actual logic


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    topo_order = []

    def dfs(var: Variable) -> None:
        """Depth-first search to traverse the computation graph."""
        if var.unique_id not in visited and not var.is_constant():
            visited.add(var.unique_id)
            for parent in var.parents:
                dfs(parent)
            topo_order.append(var)

    # Start DFS from the given variable
    dfs(variable)
    return reversed(topo_order)  # Return in reverse order for topological sorting


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Run backpropagation on the computation graph to compute derivatives.

    Args:
    ----
        variable: The right-most variable to start backpropagation from.
        deriv: The derivative of the output with respect to the variable.

    """
    # Initialize a dictionary to store the derivatives for each variable.
    derivatives = {variable.unique_id: deriv}

    # Perform topological sort to get the variables in a valid backpropagation order.
    sorted_variables = topological_sort(variable)

    # Iterate over each variable in the topologically sorted order.
    for var in sorted_variables:
        # Retrieve the derivative for this variable
        d_output = derivatives[var.unique_id]

        # If the variable is a leaf, accumulate its derivative.
        if var.is_leaf():
            var.accumulate_derivative(d_output)
        else:
            # Otherwise, propagate the gradient using the chain rule.
            for parent, local_deriv in var.chain_rule(d_output):
                if parent.unique_id in derivatives:
                    derivatives[parent.unique_id] += local_deriv
                else:
                    derivatives[parent.unique_id] = local_deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Retrieve the saved tensors for backward computation."""
        return self.saved_values

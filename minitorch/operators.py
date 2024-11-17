"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable


# Mathematical functions:
def mul(x: float, y: float) -> float:
    """Multiplies two numbers."""
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negates a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal."""
    return x == y


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float, tol: float = 1e-2) -> bool:
    """Checks if two numbers are close in value."""
    return abs(x - y) < tol


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function."""
    return max(0, x)


def log(x: float) -> float:
    """Calculates the natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function."""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal."""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Computes the derivative of log times a second argument."""
    return d / x


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of reciprocal times a second argument."""
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of ReLU times a second argument."""
    return d if x > 0 else 0


# ## Task 0.3 (Now Implemented)
# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """Applies the function `fn` to each element of `ls`."""
    return [fn(x) for x in ls]


def zipWith(
    fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> Iterable[float]:
    """Combines elements from `ls1` and `ls2` using the function `fn`."""
    return [fn(x, y) for x, y in zip(ls1, ls2)]


def reduce(
    fn: Callable[[float, float], float], start: float, ls: Iterable[float]
) -> float:
    """Reduces the iterable `ls` to a single value using the function `fn`."""
    result = start
    for x in ls:
        result = fn(result, x)
    return result


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in the list `ls` using `map`."""
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists `ls1` and `ls2` using `zipWith`."""
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in the list `ls` using `reduce`."""
    return reduce(add, 0.0, ls)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in the list `ls` using `reduce`."""
    return reduce(mul, 1.0, ls)

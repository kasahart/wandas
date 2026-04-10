"""Utilities for runtime signature introspection."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from inspect import Parameter, signature
from typing import Any, Final

__all__: Final = ["accepted_kwargs", "filter_kwargs"]


def accepted_kwargs(func: Callable[..., Any]) -> tuple[set[str], bool]:
    """
    Get the set of explicit keyword arguments accepted by
    a function and whether it accepts **kwargs.

    Args:
        func: The function to inspect.

    Returns:
        A tuple containing:
        - set[str]: Set of explicit keyword argument names accepted by func.
        - bool: Whether the function accepts variable keyword arguments (**kwargs).
    """
    # Return empty set and unlimited flag for mock objects
    if hasattr(func, "__module__") and func.__module__ == "unittest.mock":
        return set(), True
    try:
        params = signature(func).parameters.values()

        # Collect explicitly defined arguments
        explicit_kwargs = {
            p.name for p in params if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        }

        # Flag for whether the function accepts **kwargs
        has_var_kwargs = any(p.kind is Parameter.VAR_KEYWORD for p in params)

        return explicit_kwargs, has_var_kwargs
    except (ValueError, TypeError):
        # Return empty set and unlimited flag when signature cannot be obtained
        return set(), True


def filter_kwargs(
    func: Callable[..., Any],
    kwargs: Mapping[str, Any],
    *,
    strict_mode: bool = False,
) -> dict[str, Any]:
    """
    Filter keyword arguments to only those accepted by the function.

    This function examines the signature of `func` and returns a dictionary
    containing only the key-value pairs from `kwargs` that are valid keyword
    arguments for `func`.

    Args:
        func: The function to filter keyword arguments for.
        kwargs: The keyword arguments to filter.
        strict_mode: If True, only explicitly defined parameters are passed even when
            the function accepts **kwargs. If False (default), all parameters are
            passed to functions that accept **kwargs, but a warning is issued for
            parameters not explicitly defined.

    Returns:
        A dictionary containing only the key-value pairs that are valid for `func`.
    """
    explicit_params, accepts_var_kwargs = accepted_kwargs(func)

    # When **kwargs is not accepted or strict_mode is True,
    # filter to only explicitly defined parameters
    if not accepts_var_kwargs or strict_mode:
        filtered = {k: v for k, v in kwargs.items() if k in explicit_params}
        return filtered

    # When **kwargs is accepted (strict_mode is False), allow all keys
    # but warn for keys not explicitly defined
    unknown = set(kwargs) - explicit_params
    if unknown:
        warnings.warn(
            f"Implicit kwargs for {getattr(func, '__name__', repr(func))}: {unknown}",
            UserWarning,
            stacklevel=2,
        )
    return dict(kwargs)

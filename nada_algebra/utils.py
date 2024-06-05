"""General utils functions"""

from typing import Callable


def copy_metadata(source_func: Callable) -> Callable:
    """
    Copies metadata (docstring and type annotation) from one function
    and adds it to the function it decorates.

    Args:
        source_func (Callable): Source function to copy metadata from.

    Returns:
        Callable: Decorated function.
    """

    def decorator(func: Callable) -> Callable:
        """
        Decorates function with source function's metadata.

        Args:
            func (Callable): Function without added metadata.

        Returns:
            Callable: Function with metadata.
        """
        func.__doc__ = source_func.__doc__ if hasattr(source_func, "__doc__") else ""
        func.__annotations__ = (
            source_func.__annotations__
            if hasattr(source_func, "__annotations__")
            else {}
        )
        return func

    return decorator

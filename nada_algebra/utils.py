"""General utils functions"""

import re
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
        doc: str = ""
        if hasattr(source_func, "__doc__") and source_func.__doc__ is not None:
            doc = source_func.__doc__
            # Replace NumPy references with NadaArray references
            doc = re.sub(r"\b(numpy)\b", "NadaArray", doc, flags=re.IGNORECASE)
            doc = re.sub(r"\b(np)\b", "na", doc, flags=re.IGNORECASE)

        func.__doc__ = doc

        annot = {}
        if hasattr(source_func, "__annotations__"):
            annot = source_func.__annotations__
        func.__annotations__ = annot

        return func

    return decorator

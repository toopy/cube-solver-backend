"""FastAPI dependencies shared across solver backends.

This module provides helpers to resolve which backend to use based on
request metadata (query parameter or header), with a registry fallback.
"""

from typing import Optional

from fastapi import (
    FastAPI,
    Header,
    Query,
)

from .registry import get_registry


def get_backend_name(
    app: FastAPI,
    backend: Optional[str] = Query(None, description="Backend name to use"),
    x_backend: Optional[str] = Header(None, convert_underscores=False),
) -> str:
    """Resolve the solver backend name from request metadata.

    Priority order is the ``backend`` query parameter, then the
    ``X-Backend`` header. If neither is present, falls back to the
    registry's default backend.

    Parameters
    ----------
    app : fastapi.FastAPI
        Application instance holding the backends registry.
    backend : str or None, default None
        Optional query parameter ``?backend=...`` specifying the backend.
    x_backend : str or None, default None
        Optional ``X-Backend`` header specifying the backend.

    Returns
    -------
    str
        The resolved backend name.

    Raises
    ------
    ValueError
        If the resolved name is not present in the registry.
    """
    registry = get_registry(app)
    name = backend or x_backend or registry.default_backend
    if name not in registry.backends:
        raise ValueError(f"Unknown backend: {name}")
    return name

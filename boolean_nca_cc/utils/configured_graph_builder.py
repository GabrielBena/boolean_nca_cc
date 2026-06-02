"""
Configured graph builder module.

This module provides a globally configured version of build_graph that can be
set once at the start of training and used throughout the codebase without
needing to thread configuration parameters through every function call.
"""

from collections.abc import Callable
from functools import partial

from boolean_nca_cc.utils.graph_builder import build_graph

# Global configured build_graph function
_configured_build_graph: Callable | None = None


def configure_build_graph(
    neighboring_connections: bool = False, bidirectional_edges: bool = True, **kwargs
) -> None:
    """
    Configure the global build_graph function with the specified parameters.

    This should be called once at the start of training to set up the
    graph building parameters that will be used throughout the codebase.

    Args:
        neighboring_connections: Enable neighboring connections between adjacent gates
        bidirectional_edges: Create edges in both forward and backward directions
        **kwargs: Additional default parameters for build_graph
    """
    global _configured_build_graph

    # Create configured function with partial application
    _configured_build_graph = partial(
        build_graph,
        neighboring_connections=neighboring_connections,
        bidirectional_edges=bidirectional_edges,
        **kwargs,
    )


def get_configured_build_graph() -> Callable:
    """
    Get the configured build_graph function.

    If no configuration has been set, returns a function with default parameters:
    - neighboring_connections=False
    - bidirectional_edges=True

    Returns:
        The configured build_graph function (or default if not configured)
    """
    global _configured_build_graph

    if _configured_build_graph is None:
        # Return default configuration if none has been set
        return partial(
            build_graph,
            neighboring_connections=False,
            bidirectional_edges=True,
        )

    return _configured_build_graph


def get_configured_pe_flags() -> dict:
    """Return the graph-PE / node-feature flags baked into the configured build_graph.

    Reads them off the stored ``functools.partial`` keywords so callers (e.g.
    pool initialisation, build-time assertions) can introspect what
    ``build_graph`` will emit without re-plumbing config. Falls back to
    ``build_graph``'s own defaults for any flag not explicitly configured.

    Returns:
        Dict with keys ``use_dist_pe`` (bool), ``use_rwse`` (bool), ``rwse_k`` (int).
    """
    fn = get_configured_build_graph()
    kw = getattr(fn, "keywords", {}) or {}
    return {
        "use_dist_pe": kw.get("use_dist_pe", False),
        "use_rwse": kw.get("use_rwse", False),
        "rwse_k": kw.get("rwse_k", 8),
    }


def is_configured() -> bool:
    """
    Check if build_graph has been explicitly configured.

    Returns:
        True if configure_build_graph() has been called, False if using defaults
    """
    global _configured_build_graph
    return _configured_build_graph is not None


def configured_build_graph(*args, **kwargs):
    """
    Configured build_graph function that uses the globally set configuration.

    If no configuration has been set via configure_build_graph(), this function
    will use default parameters:
    - neighboring_connections=False
    - bidirectional_edges=True

    This is a convenience function that calls get_configured_build_graph()
    and forwards all arguments to the configured function.

    Args:
        *args: Positional arguments for build_graph
        **kwargs: Keyword arguments for build_graph (these override configured defaults)

    Returns:
        jraph.GraphsTuple: The built graph
    """
    configured_fn = get_configured_build_graph()
    return configured_fn(*args, **kwargs)

"""
Positional encoding utilities for graph neural networks.

This module provides:
- ``get_positional_encoding``: scalar → sinusoidal PE (the existing primitive).
- ``compute_dag_distance_pe``: graph-distance PE — sinusoidal encoding of
  ``(dist_from_input, dist_to_output)`` on the directed DAG. Absolute, directional,
  scale-free across changes in circuit depth.
- ``compute_rwse``: Random Walk Structural Encoding — for each node, the K-vector
  of return probabilities ``[p_1(v→v), p_2(v→v), …, p_K(v→v)]`` under a uniform
  random walk on the bidirectional graph. Pure-topology signature.

Both graph PEs are wire-dependent and must be recomputed when wires change. They
are JIT-safe: shapes are static given fixed ``n_node`` and ``layer_sizes``, and
the BFS / random-walk math is implemented in pure JAX.
"""

import jax
import jax.numpy as jp


def get_positional_encoding(indices: jp.ndarray, dim: int, max_val: float = 10000.0):
    """
    Generate sinusoidal positional encodings for node positions.

    Args:
        indices: 1D array of integer positions (e.g., layer indices)
        dim: Dimension of the positional encoding vector (must be even)
        max_val: Maximum value for the denominator in frequency calculation

    Returns:
        Array of shape (len(indices), dim) containing positional encodings
    """
    if dim % 2 != 0:
        raise ValueError(f"Positional encoding dimension must be even, got {dim}")

    # Ensure indices are float for calculations
    positions = indices.astype(jp.float32)[:, None]  # Shape: [num_indices, 1]

    # Calculate frequency division term
    div_term = jp.exp(jp.arange(0, dim, 2, dtype=jp.float32) * -(jp.log(max_val) / dim))

    # Initialize positional encoding matrix
    pe = jp.zeros((indices.shape[0], dim), dtype=jp.float32)

    # Calculate sine and cosine components
    pe = pe.at[:, 0::2].set(jp.sin(positions * div_term))
    pe = pe.at[:, 1::2].set(jp.cos(positions * div_term))

    return pe


def compute_dag_distance_pe(
    forward_senders: jp.ndarray,
    forward_receivers: jp.ndarray,
    layer_indices: jp.ndarray,
    n_node: int,
    n_iterations: int,
    pe_dim: int,
    max_val: float = 10000.0,
) -> jp.ndarray:
    """Sinusoidal encoding of ``(dist_from_input, dist_to_output)`` on the directed DAG.

    Bellman-Ford-style relaxation on forward edges only:
      - ``dist_from_input``: BFS from layer-0 nodes through forward edges.
      - ``dist_to_output``: BFS from max-layer nodes through *reversed* forward edges.

    Distances saturate at ``n_iterations`` for nodes that are unreachable in the
    chosen direction (cannot occur for the strictly-layered DAGs used here, where
    one relaxation step propagates exactly one layer; the saturation cap exists
    only as a safety bound and a finite max for the sinusoidal frequencies).

    The asymmetry — input-side BFS down the DAG, output-side BFS up the DAG —
    is intentional: the two distances describe the two faces of the graph the
    model has to reason about (data flow from inputs, error flow from outputs).

    Args:
        forward_senders: Forward-edge senders [E_fwd].
        forward_receivers: Forward-edge receivers [E_fwd].
        layer_indices: Per-node layer index [N]. Used to seed BFS sources
            (``layer == 0`` for input side, ``layer == max(layer)`` for output side).
        n_node: Number of nodes (static).
        n_iterations: Number of Bellman-Ford iterations. ``len(layer_sizes)`` (i.e.
            number of gate layers + 1 for the input layer) is always sufficient and
            usually loose by one.
        pe_dim: Sinusoidal dim per side. Total output dim is ``2 * pe_dim``.
        max_val: Frequency basis (matches ``get_positional_encoding`` convention).

    Returns:
        Array of shape ``[n_node, 2 * pe_dim]``: ``[PE(dist_from_input), PE(dist_to_output)]``.
    """
    inf_val = jp.float32(n_iterations + 1)

    is_input_layer = layer_indices == 0
    max_layer = jp.max(layer_indices)
    is_output_layer = layer_indices == max_layer

    def _bfs_forward(carry, _):
        # Edge relaxation: dist[r] = min(dist[r], dist[s] + 1)
        dist = carry
        candidates = dist[forward_senders] + 1.0
        new_dist = dist.at[forward_receivers].min(candidates)
        return new_dist, None

    def _bfs_backward(carry, _):
        # Edge relaxation on reversed forward edges: dist[s] = min(dist[s], dist[r] + 1)
        dist = carry
        candidates = dist[forward_receivers] + 1.0
        new_dist = dist.at[forward_senders].min(candidates)
        return new_dist, None

    dist_in_init = jp.where(is_input_layer, 0.0, inf_val).astype(jp.float32)
    dist_in, _ = jax.lax.scan(_bfs_forward, dist_in_init, None, length=n_iterations)

    dist_out_init = jp.where(is_output_layer, 0.0, inf_val).astype(jp.float32)
    dist_out, _ = jax.lax.scan(_bfs_backward, dist_out_init, None, length=n_iterations)

    pe_in = get_positional_encoding(dist_in, pe_dim, max_val=max_val)
    pe_out = get_positional_encoding(dist_out, pe_dim, max_val=max_val)

    return jp.concatenate([pe_in, pe_out], axis=-1)


def compute_rwse(
    senders: jp.ndarray,
    receivers: jp.ndarray,
    n_node: int,
    k: int,
) -> jp.ndarray:
    """Random Walk Structural Encoding.

    For each node v, returns ``[p_1(v→v), p_2(v→v), …, p_K(v→v)]`` — the probability
    of returning to v after exactly k steps of a uniform random walk on the graph
    encoded by ``(senders, receivers)``. With bidirectional senders/receivers (the
    default in ``build_graph``) this is a walk on the symmetric adjacency.

    Captures local topological signature (degree pattern, triangles, fan-in vs.
    fan-out balance, cycle structure) entirely from the graph — no eigendecomposition.

    Args:
        senders: Edge senders [E].
        receivers: Edge receivers [E].
        n_node: Number of nodes (static).
        k: Walk length / number of return-probability features per node.

    Returns:
        Array of shape ``[n_node, k]``.
    """
    # Build dense adjacency; ``add(1.0)`` so parallel edges accumulate (won't happen
    # for clean wires but harmless if they do, and avoids losing self-loop weight).
    adj = jp.zeros((n_node, n_node), dtype=jp.float32)
    adj = adj.at[senders, receivers].add(1.0)

    # Row-normalize → transition matrix. Isolated nodes (deg == 0) get zero rows,
    # so their return-probability sequence is all zeros — a meaningful "no info" code.
    deg = adj.sum(axis=1)
    deg_inv = jp.where(deg > 0, 1.0 / deg, 0.0)
    transition = adj * deg_inv[:, None]

    diag_terms = []
    power = jp.eye(n_node, dtype=jp.float32)
    for _ in range(k):
        power = power @ transition
        diag_terms.append(jp.diag(power))

    return jp.stack(diag_terms, axis=-1)


def compute_rwse_batched(
    senders: jp.ndarray,
    receivers: jp.ndarray,
    n_node: int,
    k: int,
    chunk_size: int = 256,
) -> jp.ndarray:
    """Memory-flat batched :func:`compute_rwse` over a pool of circuits.

    ``compute_rwse`` materialises a dense ``[n_node, n_node]`` transition matrix
    and its powers. Naively ``vmap``-ing it across a large pool holds
    ``pool_size × n_node²`` floats live at once — gigabytes at ``pool_size=4096``,
    ``n_node≈450``. This helper instead drives the per-circuit computation through
    ``jax.lax.map`` with a ``batch_size``, so peak memory is bounded by
    ``chunk_size × n_node²`` regardless of pool size, while still parallelising
    within each chunk.

    Args:
        senders: Batched edge senders ``[pool_size, E]``.
        receivers: Batched edge receivers ``[pool_size, E]``.
        n_node: Number of nodes per circuit (static; constant across the pool).
        k: Walk length / RWSE feature dim.
        chunk_size: Inner vmap width — the memory/throughput knob. Smaller = flatter
            memory, larger = more parallelism.

    Returns:
        Array of shape ``[pool_size, n_node, k]``.
    """

    def _single(sr):
        s, r = sr
        return compute_rwse(s, r, n_node, k)

    return jax.lax.map(_single, (senders, receivers), batch_size=chunk_size)

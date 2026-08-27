"""
Basic UMAP visualization script for degenerate circuit solution spaces.

This script loads exploration results and creates a UMAP embedding visualization
with depth coloring (distance from root circuit). Optionally overlays exploration
graph edges to show perturbation-recovery trajectories.

Usage:
    # Basic visualization (points only)
    python experiments/visualize_umap.py --results-dir <path>
    
    # Load from checkpoint with ~200 solutions (rounds to nearest checkpoint_*_solutions)
    python experiments/visualize_umap.py --results-dir <path> --solutions 200
    
    # With exploration graph edges overlaid
    python experiments/visualize_umap.py --results-dir <path> --show-edges
    
    # With edges and cycle highlighting
    python experiments/visualize_umap.py --results-dir <path> --show-edges --highlight-cycles
"""

import argparse
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple
from collections import deque

# Suppress TensorFlow CUDA plugin registration warnings
# These occur when JAX/TensorFlow tries to register CUDA plugins multiple times
# Set before importing JAX to be effective
# TF_CPP_MIN_LOG_LEVEL: 0=all, 1=no INFO, 2=no INFO/WARNING, 3=no INFO/WARNING/ERROR
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # Suppress all TensorFlow log messages
os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_enable_xla_devices=false')

# Filter TensorFlow/XLA warnings that may still appear
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', message='.*Unable to register.*factory.*')
warnings.filterwarnings('ignore', message='.*computation placer already registered.*')

import numpy as np
import jax.numpy as jp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
try:
    from umap import UMAP
except ImportError:
    import umap.umap_ as umap
    UMAP = umap.UMAP

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from experiments.explore_degenerate_solutions import load_exploration_results

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _find_checkpoints(checkpoints_dir: Path) -> List[Tuple[int, Path]]:
    """Find all checkpoint_*_solutions folders, return [(count, path), ...] sorted by count."""
    pattern = re.compile(r"checkpoint_(\d+)_solutions")
    candidates = []
    for subdir in checkpoints_dir.iterdir():
        if subdir.is_dir():
            m = pattern.match(subdir.name)
            if m:
                candidates.append((int(m.group(1)), subdir))
    return sorted(candidates, key=lambda x: x[0])


def resolve_results_path(results_dir: Path, solutions: Optional[int] = None) -> Path:
    """
    Resolve the path to load exploration results from.

    - If solutions is None: load from results_dir (root - final results). If root has no
      solutions.npz (e.g. interrupted run), fall back to latest checkpoint.
    - If solutions is int: load from results_dir/checkpoints/checkpoint_X_solutions
      where X is the checkpoint whose solution count is closest to the requested value.
      Handles "weird" checkpoint numbers (e.g. 201, 303) by rounding to nearest.
    """
    results_dir = Path(results_dir)

    if solutions is not None:
        # Explicit checkpoint selection
        checkpoints_dir = results_dir / "checkpoints"
        if not checkpoints_dir.exists():
            raise FileNotFoundError(
                f"No checkpoints directory found at {checkpoints_dir}. "
                "Use --solutions only when exploration saved checkpoints."
            )
        candidates = _find_checkpoints(checkpoints_dir)
        if not candidates:
            raise FileNotFoundError(f"No checkpoint folders found in {checkpoints_dir}")
        best = min(candidates, key=lambda x: abs(x[0] - solutions))
        log.info(f"Requested ~{solutions} solutions, using checkpoint_{best[0]}_solutions")
        return best[1]

    # Default: root (final results)
    root_solutions = results_dir / "solutions.npz"
    if root_solutions.exists():
        return results_dir

    # Root missing (interrupted run) -> fall back to latest checkpoint
    checkpoints_dir = results_dir / "checkpoints"
    if checkpoints_dir.exists():
        candidates = _find_checkpoints(checkpoints_dir)
        if candidates:
            latest = candidates[-1]
            log.info(
                f"Root has no solutions.npz (run may have been interrupted). "
                f"Using latest checkpoint: checkpoint_{latest[0]}_solutions"
            )
            return latest[1]

    raise FileNotFoundError(
        f"No exploration results found at {results_dir}. "
        "Expected solutions.npz or checkpoints/checkpoint_*_solutions/"
    )


def compute_distances_from_root(
    root_hash: str,
    exploration_graph: dict,
    unique_solutions: dict,
) -> dict:
    """
    Compute shortest path distances from root to all solutions using BFS.
    
    Args:
        root_hash: Hash of root circuit
        exploration_graph: Graph structure (source_hash -> [(target_hash, pattern_idx, metadata)])
        unique_solutions: Dictionary of unique solutions (hash -> logits)
        
    Returns:
        Dictionary mapping circuit_hash -> distance from root
    """
    distances = {root_hash: 0}
    queue = deque([root_hash])
    
    while queue:
        current = queue.popleft()
        current_distance = distances[current]
        
        # Get all neighbors from exploration graph
        for target_hash, _, _ in exploration_graph.get(current, []):
            if target_hash not in distances:
                distances[target_hash] = current_distance + 1
                queue.append(target_hash)
    
    # Set distance to inf for unreachable circuits
    for circuit_hash in unique_solutions.keys():
        if circuit_hash not in distances:
            distances[circuit_hash] = float('inf')
    
    return distances


def prepare_feature_vectors(
    unique_solutions: dict,
    distances: dict,
) -> tuple:
    """
    Prepare feature vectors from circuit logits for UMAP.
    
    Args:
        unique_solutions: Dictionary of unique solutions (hash -> logits)
        distances: Dictionary mapping circuit_hash -> distance from root
        
    Returns:
        Tuple of (feature_matrix, circuit_hashes, depth_values)
    """
    feature_vectors = []
    circuit_hashes = []
    depth_values = []
    
    for circuit_hash, logits in unique_solutions.items():
        # Flatten all logits into a single feature vector
        flat_logits = jp.concatenate([l.flatten() for l in logits])
        feature_vectors.append(np.array(flat_logits))
        circuit_hashes.append(circuit_hash)
        depth_values.append(distances.get(circuit_hash, float('inf')))
    
    feature_matrix = np.array(feature_vectors)
    
    return feature_matrix, circuit_hashes, depth_values


def visualize_umap(
    feature_matrix: np.ndarray,
    depth_values: list,
    circuit_hashes: list,
    root_hash: str,
    output_file: Path = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "euclidean",
    cmap: str = "viridis",
    figsize: tuple = (10, 8),
    save_dir: Path = None,
    exploration_graph: dict = None,
    show_edges: bool = False,
    edge_alpha: float = 0.05,
    edge_linewidth: float = 0.5,
    edge_color: str = None,
    highlight_cycles: bool = False,
    embedding: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create UMAP embedding and visualize with depth coloring.
    
    Args:
        feature_matrix: Feature matrix (n_samples, n_features)
        depth_values: List of depth values for each sample
        circuit_hashes: List of circuit hashes for each sample
        root_hash: Hash of root circuit (for highlighting)
        output_file: Optional path to save figure
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        n_components: Number of UMAP dimensions (2 or 3)
        metric: Distance metric for UMAP
        cmap: Colormap for depth coloring
        figsize: Figure size
        save_dir: Optional directory to save UMAP results (embedding, metadata, figure)
        exploration_graph: Graph structure (source_hash -> [(target_hash, pattern_idx, metadata)])
        show_edges: Whether to overlay exploration graph edges
        edge_alpha: Transparency for edges (0-1)
        edge_linewidth: Line width for edges
        edge_color: Color for edges (default: "gray", or specify color)
        highlight_cycles: Whether to highlight self-recovery cycles (A -> A)
        
    Returns:
        UMAP embedding array
    """
    if embedding is None:
        log.info(f"Running UMAP on {len(feature_matrix)} circuits...")
        log.info(f"  Feature matrix shape: {feature_matrix.shape}")
        log.info(f"  UMAP parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
        reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
        )
        embedding = reducer.fit_transform(feature_matrix)
        log.info(f"  Embedding shape: {embedding.shape}")
    else:
        log.info(f"Using precomputed UMAP embedding (shape: {embedding.shape})")
    
    # Save results if save_dir is provided
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"\nSaving UMAP results to: {save_dir}")
        
        # Save embedding
        np.save(save_dir / "embedding.npy", embedding)
        log.info(f"  Saved embedding.npy")
        
        # Save feature matrix
        np.save(save_dir / "feature_matrix.npy", feature_matrix)
        log.info(f"  Saved feature_matrix.npy")
        
        # Save depth values
        np.save(save_dir / "depth_values.npy", np.array(depth_values))
        log.info(f"  Saved depth_values.npy")
        
        # Save circuit hashes as text file (one per line)
        with open(save_dir / "circuit_hashes.txt", "w") as f:
            f.write("\n".join(circuit_hashes))
        log.info(f"  Saved circuit_hashes.txt")
    
    # Convert depth values to array
    depth_array = np.array(depth_values)
    
    # Find root circuit index
    root_idx = None
    for idx, circuit_hash in enumerate(circuit_hashes):
        if circuit_hash == root_hash:
            root_idx = idx
            break
    
    # Create hash-to-index mapping for edge visualization
    hash_to_idx = {hash_val: idx for idx, hash_val in enumerate(circuit_hashes)}
    
    # Create visualization
    if n_components == 2:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw edges first (background) so points render on top
        if show_edges and exploration_graph is not None:
            edges_drawn = 0
            cycles_drawn = 0
            
            for source_hash, edges_list in exploration_graph.items():
                source_idx = hash_to_idx.get(source_hash)
                if source_idx is None:
                    continue
                
                for target_hash, pattern_idx, metadata in edges_list:
                    target_idx = hash_to_idx.get(target_hash)
                    if target_idx is None:
                        continue
                    
                    # Skip self-loops unless highlighting cycles
                    is_cycle = (source_hash == target_hash)
                    if is_cycle and not highlight_cycles:
                        continue
                    
                    # Determine edge color
                    color = edge_color if edge_color is not None else "gray"
                    
                    # Highlight cycles with different style
                    if is_cycle:
                        ax.plot(
                            [embedding[source_idx, 0], embedding[target_idx, 0]],
                            [embedding[source_idx, 1], embedding[target_idx, 1]],
                            color=color,
                            alpha=min(edge_alpha * 2, 1.0),  # More visible for cycles
                            linewidth=edge_linewidth * 2,
                            linestyle='--',
                            zorder=0,
                        )
                        cycles_drawn += 1
                    else:
                        ax.plot(
                            [embedding[source_idx, 0], embedding[target_idx, 0]],
                            [embedding[source_idx, 1], embedding[target_idx, 1]],
                            color=color,
                            alpha=edge_alpha,
                            linewidth=edge_linewidth,
                            zorder=0,
                        )
                        edges_drawn += 1
            
            log.info(f"  Drawn {edges_drawn} edges, {cycles_drawn} cycles")
        
        # Scatter plot with depth coloring (foreground, on top of edges)
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=depth_array,
            cmap=cmap,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidths=0.5,
            zorder=2,
        )
        
        # Highlight root circuit
        if root_idx is not None:
            ax.scatter(
                embedding[root_idx, 0],
                embedding[root_idx, 1],
                c='red',
                marker='*',
                s=500,
                edgecolors='black',
                linewidths=2,
                label='Root circuit',
                zorder=10,
            )
        
        ax.set_xlabel('UMAP Dimension 1', fontsize=18)
        ax.set_ylabel('UMAP Dimension 2', fontsize=18)
        ax.tick_params(axis='both', labelsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Perturbations from Root', fontsize=18)
        cbar.ax.tick_params(labelsize=10)
        
        if root_idx is not None:
            ax.legend(loc='best', fontsize=16)
        
        plt.tight_layout()
        
    elif n_components == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw edges first (background) so points render on top
        if show_edges and exploration_graph is not None:
            edges_drawn = 0
            cycles_drawn = 0
            
            for source_hash, edges_list in exploration_graph.items():
                source_idx = hash_to_idx.get(source_hash)
                if source_idx is None:
                    continue
                
                for target_hash, pattern_idx, metadata in edges_list:
                    target_idx = hash_to_idx.get(target_hash)
                    if target_idx is None:
                        continue
                    
                    # Skip self-loops unless highlighting cycles
                    is_cycle = (source_hash == target_hash)
                    if is_cycle and not highlight_cycles:
                        continue
                    
                    # Determine edge color
                    color = edge_color if edge_color is not None else "gray"
                    
                    # Highlight cycles with different style
                    if is_cycle:
                        ax.plot(
                            [embedding[source_idx, 0], embedding[target_idx, 0]],
                            [embedding[source_idx, 1], embedding[target_idx, 1]],
                            [embedding[source_idx, 2], embedding[target_idx, 2]],
                            color=color,
                            alpha=min(edge_alpha * 2, 1.0),  # More visible for cycles
                            linewidth=edge_linewidth * 2,
                            linestyle='--',
                            zorder=0,
                        )
                        cycles_drawn += 1
                    else:
                        ax.plot(
                            [embedding[source_idx, 0], embedding[target_idx, 0]],
                            [embedding[source_idx, 1], embedding[target_idx, 1]],
                            [embedding[source_idx, 2], embedding[target_idx, 2]],
                            color=color,
                            alpha=edge_alpha,
                            linewidth=edge_linewidth,
                            zorder=0,
                        )
                        edges_drawn += 1
            
            log.info(f"  Drawn {edges_drawn} edges, {cycles_drawn} cycles")
        
        # 3D scatter plot with depth coloring (foreground, on top of edges)
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            c=depth_array,
            cmap=cmap,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidths=0.5,
            zorder=2,
        )
        
        # Highlight root circuit
        if root_idx is not None:
            ax.scatter(
                embedding[root_idx, 0],
                embedding[root_idx, 1],
                embedding[root_idx, 2],
                c='red',
                marker='*',
                s=500,
                edgecolors='black',
                linewidths=2,
                label='Root circuit',
            )
        
        title_suffix = ""
        if show_edges:
            title_suffix = " (with Exploration Trajectories)"
        
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_zlabel('UMAP Dimension 3', fontsize=12)
        ax.set_title(f'UMAP Embedding of Circuit Solutions\n(Colored by Perturbations from Root){title_suffix}', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Perturbations from Root', fontsize=12)
        
        # Add legend
        if root_idx is not None:
            ax.legend(loc='best')
    
    # Save figure
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
        log.info(f"Saved figure to: {output_file}")
    elif save_dir is not None:
        # Auto-save figure to save_dir if no explicit output_file
        figure_path = save_dir / "figure.png"
        plt.savefig(figure_path, dpi=600, bbox_inches='tight')
        log.info(f"  Saved figure.png")
    
    if output_file is None:
        plt.show()
    
    # Print statistics
    log.info("\n" + "=" * 80)
    log.info("UMAP Visualization Statistics")
    log.info("=" * 80)
    log.info(f"Total circuits: {len(circuit_hashes)}")
    log.info(f"Depth distribution:")
    unique_depths, counts = np.unique(depth_array[depth_array != np.inf], return_counts=True)
    for depth, count in zip(unique_depths, counts):
        log.info(f"  Depth {int(depth)}: {count} circuits")
    unreachable = np.sum(depth_array == np.inf)
    if unreachable > 0:
        log.info(f"  Unreachable: {unreachable} circuits")
    log.info("=" * 80)
    
    return embedding


def build_depth_frames(
    exploration_results: list,
    root_hash: str,
    distances: dict,
) -> Tuple[List[set], List[List[Tuple[str, str, bool]]], int]:
    """
    Group exploration history into per-depth cumulative frames.

    Frame k contains the root plus every circuit whose BFS distance from
    the root is <= k. Edges enter at frame `source_depth + 1` (i.e. the
    frame in which the target first becomes visible).

    Args:
        exploration_results: Ordered list of perturbation-recovery records.
        root_hash: Hash of the root circuit (visible from frame 0).
        distances: Mapping circuit_hash -> BFS distance from root.

    Returns:
        (nodes_per_frame, edges_per_frame, max_depth)
        - nodes_per_frame[k]: cumulative set of visible circuit hashes at frame k
        - edges_per_frame[k]: cumulative list of (source, target, is_cycle) at frame k
        - max_depth: index of the final frame
    """
    finite_depths = [d for d in distances.values() if d != float('inf')]
    max_depth = int(max(finite_depths)) if finite_depths else 0

    nodes_per_frame: List[set] = [set() for _ in range(max_depth + 1)]
    edges_per_frame: List[List[Tuple[str, str, bool]]] = [[] for _ in range(max_depth + 1)]

    # Seed frame 0 with the root.
    nodes_per_frame[0].add(root_hash)

    # Place each node at its depth frame.
    for circuit_hash, d in distances.items():
        if d == float('inf'):
            continue
        di = int(d)
        if 0 <= di <= max_depth:
            nodes_per_frame[di].add(circuit_hash)

    # Place each edge at the frame where its target becomes visible.
    # Deduplicate by (source, target) so revisits don't double-count.
    seen_edges = set()
    for rec in exploration_results:
        src = rec.get("source_hash")
        tgt = rec.get("recovered_hash")
        if src is None or tgt is None:
            continue
        if (src, tgt) in seen_edges:
            continue
        seen_edges.add((src, tgt))

        src_d = distances.get(src, float('inf'))
        tgt_d = distances.get(tgt, float('inf'))
        if src_d == float('inf') or tgt_d == float('inf'):
            continue
        # Edge enters when the target appears; for self-loops that's the
        # source's own depth.
        enter_frame = int(tgt_d) if src != tgt else int(src_d)
        if 0 <= enter_frame <= max_depth:
            edges_per_frame[enter_frame].append((src, tgt, src == tgt))

    # Make cumulative.
    for k in range(1, max_depth + 1):
        nodes_per_frame[k] |= nodes_per_frame[k - 1]
        edges_per_frame[k] = edges_per_frame[k - 1] + edges_per_frame[k]

    return nodes_per_frame, edges_per_frame, max_depth


def animate_umap(
    embedding: np.ndarray,
    depth_values: list,
    circuit_hashes: list,
    root_hash: str,
    exploration_results: list,
    distances: dict,
    output_path: Path,
    cmap: str = "viridis",
    figsize: tuple = (10, 8),
    fps: int = 4,
    hold_frames: int = 6,
    show_edges: bool = True,
    edge_alpha: float = 0.08,
    edge_linewidth: float = 0.5,
    edge_color: str = None,
    highlight_cycles: bool = False,
) -> None:
    """
    Render an animated GIF of the UMAP embedding, revealing one BFS depth
    layer per frame.

    The embedding is precomputed (positions are fixed across frames); only
    visibility of points and edges changes per frame.
    """
    if embedding.shape[1] != 2:
        raise ValueError(
            f"animate_umap requires a 2D embedding (got {embedding.shape[1]}D)."
        )

    nodes_per_frame, edges_per_frame, max_depth = build_depth_frames(
        exploration_results=exploration_results,
        root_hash=root_hash,
        distances=distances,
    )

    log.info(f"\nBuilding animation: {max_depth + 1} depth frames "
             f"(+ {hold_frames} hold frames), fps={fps}")

    hash_to_idx = {h: i for i, h in enumerate(circuit_hashes)}
    depth_array = np.array(depth_values, dtype=float)
    finite_depths = depth_array[np.isfinite(depth_array)]
    vmin = float(finite_depths.min()) if finite_depths.size else 0.0
    vmax = float(finite_depths.max()) if finite_depths.size else 1.0

    # Fixed axes limits with a small margin so the camera doesn't jump.
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

    fig, ax = plt.subplots(figsize=figsize)

    # Build the colorbar once using a dummy mappable so it stays put across frames.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Perturbations from Root', fontsize=18)
    cbar.ax.tick_params(labelsize=10)

    edge_col = edge_color if edge_color is not None else "gray"
    total_frames = (max_depth + 1) + max(0, hold_frames)

    def draw_frame(frame_idx: int) -> None:
        # Clamp held frames to the last real depth.
        k = min(frame_idx, max_depth)
        ax.clear()
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel('UMAP Dimension 1', fontsize=18)
        ax.set_ylabel('UMAP Dimension 2', fontsize=18)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_title(f'Depth {k} / {max_depth}', fontsize=16)

        # Edges (background).
        if show_edges:
            for src, tgt, is_cycle in edges_per_frame[k]:
                if is_cycle and not highlight_cycles:
                    continue
                si = hash_to_idx.get(src)
                ti = hash_to_idx.get(tgt)
                if si is None or ti is None:
                    continue
                if is_cycle:
                    ax.plot(
                        [embedding[si, 0], embedding[ti, 0]],
                        [embedding[si, 1], embedding[ti, 1]],
                        color=edge_col,
                        alpha=min(edge_alpha * 2, 1.0),
                        linewidth=edge_linewidth * 2,
                        linestyle='--',
                        zorder=0,
                    )
                else:
                    ax.plot(
                        [embedding[si, 0], embedding[ti, 0]],
                        [embedding[si, 1], embedding[ti, 1]],
                        color=edge_col,
                        alpha=edge_alpha,
                        linewidth=edge_linewidth,
                        zorder=0,
                    )

        # Visible points.
        visible_idx = [hash_to_idx[h] for h in nodes_per_frame[k] if h in hash_to_idx]
        if visible_idx:
            visible_idx = np.array(visible_idx)
            ax.scatter(
                embedding[visible_idx, 0],
                embedding[visible_idx, 1],
                c=depth_array[visible_idx],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidths=0.5,
                zorder=2,
            )

        # Root marker on top.
        root_idx = hash_to_idx.get(root_hash)
        if root_idx is not None:
            ax.scatter(
                embedding[root_idx, 0],
                embedding[root_idx, 1],
                c='red',
                marker='*',
                s=500,
                edgecolors='black',
                linewidths=2,
                label='Root circuit',
                zorder=10,
            )
            ax.legend(loc='best', fontsize=14)

    anim = FuncAnimation(
        fig,
        draw_frame,
        frames=total_frames,
        interval=1000 / max(fps, 1),
        blit=False,
        repeat=True,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    log.info(f"  Saved animation: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize degenerate circuit solutions using UMAP"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="exploration_results/exploration_20251113_152000",
        help="Path to exploration results directory (default: exploration_results/exploration_20251113_152000)",
    )
    parser.add_argument(
        "--solutions",
        type=int,
        default=None,
        metavar="N",
        help="Load from checkpoint with ~N solutions (rounds to nearest checkpoint_*_solutions). "
        "If omitted, load from root (final results).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save figure (default: show interactively)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter (default: 15)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (default: 0.1)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of UMAP dimensions (default: 2)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="Distance metric for UMAP (default: euclidean)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap for depth coloring (default: viridis)",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="10,8",
        help="Figure size as 'width,height' (default: 10,8)",
    )
    parser.add_argument(
        "--show-edges",
        action="store_true",
        help="Overlay exploration graph edges on UMAP visualization",
    )
    parser.add_argument(
        "--edge-alpha",
        type=float,
        default=0.08,
        help="Transparency for edges (0-1, default: 0.08)",
    )
    parser.add_argument(
        "--edge-linewidth",
        type=float,
        default=0.5,
        help="Line width for edges (default: 0.5)",
    )
    parser.add_argument(
        "--edge-color",
        type=str,
        default=None,
        help="Color for edges (default: gray)",
    )
    parser.add_argument(
        "--highlight-cycles",
        action="store_true",
        help="Highlight self-recovery cycles (A -> A) with dashed lines",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Also produce an animated GIF revealing one BFS depth layer per frame "
        "(2D only; saved alongside the static figure).",
    )
    parser.add_argument(
        "--animation-fps",
        type=int,
        default=4,
        help="Frames per second for the animation (default: 4).",
    )
    parser.add_argument(
        "--animation-hold-frames",
        type=int,
        default=6,
        help="Extra repeats of the final frame so the GIF pauses before looping (default: 6).",
    )

    args = parser.parse_args()

    if args.animate and args.n_components != 2:
        parser.error("--animate is only supported for --n-components 2")
    
    # Parse figure size
    try:
        figsize = tuple(map(float, args.figsize.split(',')))
    except:
        log.warning(f"Invalid figsize '{args.figsize}', using default (10, 8)")
        figsize = (10, 8)
    
    # Resolve path: root or nearest checkpoint by solution count
    results_dir = Path(args.results_dir)
    load_path = resolve_results_path(results_dir, args.solutions)
    log.info(f"Loading exploration results from: {load_path}")
    results = load_exploration_results(load_path)
    num_solutions = len(results["unique_solutions"])

    # Derive UMAP visualization directory from exploration directory name + solution count
    # e.g., DFS_10_4_ROOT_SA_8zzudzmv_20260211_085457 -> umap_viz_DFS_10_4_ROOT_SA_8zzudzmv_20260211_085457_1108_solutions
    # When --output-file is set (e.g. parameter sweep), skip saving artifacts to avoid overwriting.
    exploration_dir_name = results_dir.name
    if exploration_dir_name.startswith("exploration_"):
        base_name = exploration_dir_name.replace("exploration_", "umap_viz_", 1)
    else:
        base_name = f"umap_viz_{exploration_dir_name}"
    umap_viz_dir = None if args.output_file else (results_dir.parent / f"{base_name}_{num_solutions}_solutions")

    log.info(f"Loaded {num_solutions} unique solutions")
    log.info(f"Exploration graph has {len(results['exploration_graph'])} nodes")
    log.info(f"Total edges: {len(results['edges'])}")
    
    # Compute distances from root
    log.info("\nComputing distances from root circuit...")
    distances = compute_distances_from_root(
        root_hash=results["root_hash"],
        exploration_graph=results["exploration_graph"],
        unique_solutions=results["unique_solutions"],
    )
    
    # Prepare feature vectors
    log.info("\nPreparing feature vectors...")
    feature_matrix, circuit_hashes, depth_values = prepare_feature_vectors(
        unique_solutions=results["unique_solutions"],
        distances=distances,
    )
    
    # If animating, precompute the embedding once and reuse it for both the
    # static figure and the GIF so the layouts match.
    precomputed_embedding = None
    if args.animate:
        log.info(f"\nFitting UMAP once for static + animation (n={len(feature_matrix)})...")
        reducer = UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            n_components=args.n_components,
            metric=args.metric,
        )
        precomputed_embedding = reducer.fit_transform(feature_matrix)

    # Create visualization
    log.info("\nCreating UMAP visualization...")
    embedding = visualize_umap(
        feature_matrix=feature_matrix,
        depth_values=depth_values,
        circuit_hashes=circuit_hashes,
        root_hash=results["root_hash"],
        output_file=args.output_file,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        metric=args.metric,
        cmap=args.cmap,
        figsize=figsize,
        save_dir=umap_viz_dir,  # None when --output-file is set (sweep mode)
        exploration_graph=results["exploration_graph"] if args.show_edges else None,
        show_edges=args.show_edges,
        edge_alpha=args.edge_alpha,
        edge_linewidth=args.edge_linewidth,
        edge_color=args.edge_color,
        highlight_cycles=args.highlight_cycles,
        embedding=precomputed_embedding,
    )

    if args.animate:
        # Save animation alongside other UMAP artifacts; fall back to the
        # exploration directory when no save_dir was derived (sweep mode).
        anim_dir = umap_viz_dir if umap_viz_dir is not None else results_dir
        anim_path = Path(anim_dir) / "animation.gif"
        log.info(f"\nRendering animation to {anim_path}...")
        animate_umap(
            embedding=embedding,
            depth_values=depth_values,
            circuit_hashes=circuit_hashes,
            root_hash=results["root_hash"],
            exploration_results=results["exploration_results"],
            distances=distances,
            output_path=anim_path,
            cmap=args.cmap,
            figsize=figsize,
            fps=args.animation_fps,
            hold_frames=args.animation_hold_frames,
            show_edges=args.show_edges,
            edge_alpha=args.edge_alpha,
            edge_linewidth=args.edge_linewidth,
            edge_color=args.edge_color,
            highlight_cycles=args.highlight_cycles,
        )

    if umap_viz_dir is not None:
        log.info(f"\nUMAP results saved to: {umap_viz_dir}")


if __name__ == "__main__":
    main()


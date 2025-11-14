"""
Basic UMAP visualization script for degenerate circuit solution spaces.

This script loads exploration results and creates a UMAP embedding visualization
with depth coloring (distance from root circuit). Optionally overlays exploration
graph edges to show perturbation-recovery trajectories.

Usage:
    # Basic visualization (points only)
    python experiments/visualize_umap.py --results-dir <path>
    
    # With exploration graph edges overlaid
    python experiments/visualize_umap.py --results-dir <path> --show-edges
    
    # With edges and cycle highlighting
    python experiments/visualize_umap.py --results-dir <path> --show-edges --highlight-cycles
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
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
    edge_alpha: float = 0.2,
    edge_linewidth: float = 0.5,
    edge_color: str = None,
    highlight_cycles: bool = False,
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
    log.info(f"Running UMAP on {len(feature_matrix)} circuits...")
    log.info(f"  Feature matrix shape: {feature_matrix.shape}")
    log.info(f"  UMAP parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
    
    # Create UMAP reducer
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
    )
    
    # Fit and transform
    embedding = reducer.fit_transform(feature_matrix)
    log.info(f"  Embedding shape: {embedding.shape}")
    
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
        
        # Scatter plot with depth coloring
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=depth_array,
            cmap=cmap,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidths=0.5,
        )
        
        # Overlay exploration graph edges if requested
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
                            zorder=1,
                        )
                        cycles_drawn += 1
                    else:
                        ax.plot(
                            [embedding[source_idx, 0], embedding[target_idx, 0]],
                            [embedding[source_idx, 1], embedding[target_idx, 1]],
                            color=color,
                            alpha=edge_alpha,
                            linewidth=edge_linewidth,
                            zorder=1,
                        )
                        edges_drawn += 1
            
            log.info(f"  Drawn {edges_drawn} edges, {cycles_drawn} cycles")
        
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
        
        title_suffix = ""
        if show_edges:
            title_suffix = " (with Exploration Trajectories)"
        
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_title(f'UMAP Embedding of Circuit Solutions\n(Colored by Distance from Root){title_suffix}', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Distance from Root', fontsize=12)
        
        # Add legend
        if root_idx is not None:
            ax.legend(loc='best')
        
        plt.tight_layout()
        
    elif n_components == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 3D scatter plot with depth coloring
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
        )
        
        # Overlay exploration graph edges if requested
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
                        )
                        edges_drawn += 1
            
            log.info(f"  Drawn {edges_drawn} edges, {cycles_drawn} cycles")
        
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
        ax.set_title(f'UMAP Embedding of Circuit Solutions\n(Colored by Distance from Root){title_suffix}', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Distance from Root', fontsize=12)
        
        # Add legend
        if root_idx is not None:
            ax.legend(loc='best')
    
    # Save figure
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        log.info(f"Saved figure to: {output_file}")
    elif save_dir is not None:
        # Auto-save figure to save_dir if no explicit output_file
        figure_path = save_dir / "figure.png"
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        log.info(f"  Saved figure.png")
    
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
        default=0.2,
        help="Transparency for edges (0-1, default: 0.2)",
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
    
    args = parser.parse_args()
    
    # Parse figure size
    try:
        figsize = tuple(map(float, args.figsize.split(',')))
    except:
        log.warning(f"Invalid figsize '{args.figsize}', using default (10, 8)")
        figsize = (10, 8)
    
    # Load exploration results
    results_dir = Path(args.results_dir)
    log.info(f"Loading exploration results from: {results_dir}")
    results = load_exploration_results(results_dir)
    
    # Derive UMAP visualization directory from exploration directory name
    # e.g., exploration_20251113_152000 -> umap_viz_20251113_152000
    exploration_dir_name = results_dir.name
    if exploration_dir_name.startswith("exploration_"):
        umap_viz_dir = results_dir.parent / exploration_dir_name.replace("exploration_", "umap_viz_", 1)
    else:
        umap_viz_dir = results_dir.parent / f"umap_viz_{exploration_dir_name}"
    
    log.info(f"Loaded {len(results['unique_solutions'])} unique solutions")
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
    
    # Create visualization
    log.info("\nCreating UMAP visualization...")
    visualize_umap(
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
        save_dir=umap_viz_dir,
        exploration_graph=results["exploration_graph"] if args.show_edges else None,
        show_edges=args.show_edges,
        edge_alpha=args.edge_alpha,
        edge_linewidth=args.edge_linewidth,
        edge_color=args.edge_color,
        highlight_cycles=args.highlight_cycles,
    )
    
    log.info(f"\nUMAP results saved to: {umap_viz_dir}")


if __name__ == "__main__":
    main()


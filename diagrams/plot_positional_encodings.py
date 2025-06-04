#!/usr/bin/env python3
"""
Script to visualize positional encodings in a Boolean Circuit Graph.

This script creates a graph from the main config and visualizes both
layer positional encodings and intra-layer positional encodings.
"""

import os
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc import generate_layer_sizes
from boolean_nca_cc.utils.graph_builder import build_graph


def plot_positional_encodings(graph, output_dir="plots"):
    """
    Plot the positional encodings from a graph.

    Args:
        graph: jraph.GraphsTuple containing the circuit graph
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract node features
    nodes = graph.nodes
    layer_pe = np.array(nodes["layer_pe"])  # Shape: [n_nodes, pe_dim]
    intra_layer_pe = np.array(nodes["intra_layer_pe"])  # Shape: [n_nodes, pe_dim]
    layer_indices = np.array(nodes["layer"])  # Which layer each node belongs to
    gate_ids = np.array(nodes["gate_id"])  # Global gate IDs

    n_nodes, pe_dim = layer_pe.shape
    unique_layers = np.unique(layer_indices)

    print(
        f"Graph contains {n_nodes} nodes with {pe_dim}-dimensional positional encodings"
    )
    print(f"Layers: {unique_layers}")

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Layer Positional Encoding Heatmap
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(layer_pe.T, cmap="viridis", aspect="auto")
    ax1.set_title(
        "Layer Positional Encoding\n(per node)", fontsize=14, fontweight="bold"
    )
    ax1.set_xlabel("Node Index")
    ax1.set_ylabel("PE Dimension")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # 2. Intra-Layer Positional Encoding Heatmap
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(intra_layer_pe.T, cmap="plasma", aspect="auto")
    ax2.set_title(
        "Intra-Layer Positional Encoding\n(per node)", fontsize=14, fontweight="bold"
    )
    ax2.set_xlabel("Node Index")
    ax2.set_ylabel("PE Dimension")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # 3. Layer PE grouped by actual layers
    ax3 = plt.subplot(2, 3, 3)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_layers)))
    for i, layer_idx in enumerate(unique_layers):
        mask = layer_indices == layer_idx
        layer_nodes = np.where(mask)[0]
        if len(layer_nodes) > 0:
            # Plot the mean PE for this layer
            mean_pe = np.mean(layer_pe[mask], axis=0)
            ax3.plot(
                mean_pe,
                color=colors[i],
                linewidth=2,
                label=f"Layer {layer_idx} (n={np.sum(mask)})",
            )

    ax3.set_title("Layer PE: Average per Layer", fontsize=14, fontweight="bold")
    ax3.set_xlabel("PE Dimension")
    ax3.set_ylabel("Average PE Value")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Intra-Layer PE for each layer separately
    ax4 = plt.subplot(2, 3, 4)
    for i, layer_idx in enumerate(unique_layers):
        mask = layer_indices == layer_idx
        if np.sum(mask) > 0:
            intra_pe_layer = intra_layer_pe[mask]
            # Plot first few dimensions of intra-layer PE
            for dim in range(min(5, pe_dim)):
                ax4.scatter(
                    np.where(mask)[0],
                    intra_pe_layer[:, dim],
                    color=colors[i],
                    alpha=0.6,
                    s=20,
                    label=f"L{layer_idx} dim{dim}" if dim < 3 else "",
                )

    ax4.set_title("Intra-Layer PE: First 5 Dimensions", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Node Index")
    ax4.set_ylabel("PE Value")
    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax4.grid(True, alpha=0.3)

    # 5. PE similarity matrix (cosine similarity)
    ax5 = plt.subplot(2, 3, 5)
    # Compute cosine similarity between layer PEs
    layer_pe_norm = layer_pe / (np.linalg.norm(layer_pe, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = layer_pe_norm @ layer_pe_norm.T

    im5 = ax5.imshow(similarity_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax5.set_title("Layer PE Cosine Similarity", fontsize=14, fontweight="bold")
    ax5.set_xlabel("Node Index")
    ax5.set_ylabel("Node Index")
    plt.colorbar(im5, ax=ax5, shrink=0.8)

    # Add layer boundaries
    layer_boundaries = []
    current_pos = 0
    for layer_idx in unique_layers:
        layer_size = np.sum(layer_indices == layer_idx)
        if current_pos > 0:
            layer_boundaries.append(current_pos)
        current_pos += layer_size

    for boundary in layer_boundaries:
        ax5.axhline(y=boundary - 0.5, color="white", linewidth=2, alpha=0.8)
        ax5.axvline(x=boundary - 0.5, color="white", linewidth=2, alpha=0.8)

    # 6. 2D PCA visualization of positional encodings
    ax6 = plt.subplot(2, 3, 6)
    from sklearn.decomposition import PCA

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    layer_pe_2d = pca.fit_transform(layer_pe)

    for i, layer_idx in enumerate(unique_layers):
        mask = layer_indices == layer_idx
        ax6.scatter(
            layer_pe_2d[mask, 0],
            layer_pe_2d[mask, 1],
            color=colors[i],
            alpha=0.7,
            s=50,
            label=f"Layer {layer_idx}",
        )

        # Add text annotations for some nodes
        if np.sum(mask) <= 10:  # Only annotate if not too many nodes
            for idx in np.where(mask)[0]:
                ax6.annotate(
                    f"{idx}",
                    (layer_pe_2d[idx, 0], layer_pe_2d[idx, 1]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.7,
                )

    ax6.set_title(
        f"Layer PE - 2D PCA\n(explained var: {pca.explained_variance_ratio_.sum():.3f})",
        fontsize=14,
        fontweight="bold",
    )
    ax6.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.3f})")
    ax6.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.3f})")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, "positional_encodings.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Positional encoding plots saved to: {output_path}")

    # Also create individual plots for better detail
    create_detailed_plots(graph, output_dir)

    plt.show()


def create_detailed_plots(graph, output_dir):
    """Create individual detailed plots for each type of positional encoding."""

    nodes = graph.nodes
    layer_pe = np.array(nodes["layer_pe"])
    intra_layer_pe = np.array(nodes["intra_layer_pe"])
    layer_indices = np.array(nodes["layer"])
    unique_layers = np.unique(layer_indices)

    # Detailed Layer PE plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_layers)))

    for i, layer_idx in enumerate(unique_layers):
        mask = layer_indices == layer_idx
        layer_nodes = np.where(mask)[0]

        if len(layer_nodes) > 0:
            # Plot individual nodes with transparency
            for node_idx in layer_nodes:
                plt.plot(layer_pe[node_idx], alpha=0.3, color=colors[i], linewidth=1)

            # Plot mean with thick line
            mean_pe = np.mean(layer_pe[mask], axis=0)
            plt.plot(
                mean_pe,
                color=colors[i],
                linewidth=3,
                label=f"Layer {layer_idx} (n={np.sum(mask)})",
            )

    plt.title(
        "Layer Positional Encodings - Detailed View", fontsize=16, fontweight="bold"
    )
    plt.xlabel("Positional Encoding Dimension")
    plt.ylabel("PE Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "layer_pe_detailed.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    # Detailed Intra-Layer PE plot
    plt.figure(figsize=(15, 10))
    n_layers = len(unique_layers)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    for i, layer_idx in enumerate(unique_layers):
        plt.subplot(n_rows, n_cols, i + 1)
        mask = layer_indices == layer_idx
        layer_intra_pe = intra_layer_pe[mask]

        if layer_intra_pe.shape[0] > 0:
            # Heatmap of intra-layer PE for this layer
            im = plt.imshow(layer_intra_pe.T, cmap="viridis", aspect="auto")
            plt.title(f"Layer {layer_idx} Intra-PE\n({np.sum(mask)} nodes)")
            plt.xlabel("Node within Layer")
            plt.ylabel("PE Dimension")
            plt.colorbar(im, shrink=0.8)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "intra_layer_pe_detailed.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def print_graph_info(graph):
    """Print detailed information about the graph structure."""
    nodes = graph.nodes
    n_nodes = int(graph.n_node[0])
    n_edges = int(graph.n_edge[0])

    print(f"\n=== Graph Structure ===")
    print(f"Total nodes: {n_nodes}")
    print(f"Total edges: {n_edges}")

    layer_indices = np.array(nodes["layer"])
    unique_layers, layer_counts = np.unique(layer_indices, return_counts=True)

    print(f"\nNodes per layer:")
    for layer, count in zip(unique_layers, layer_counts):
        layer_type = "Input" if layer == 0 else f"Gate Layer {layer}"
        print(f"  {layer_type}: {count} nodes")

    print(f"\nNode feature shapes:")
    for key, value in nodes.items():
        print(f"  {key}: {value.shape}")

    print(f"\nGlobal features: {graph.globals}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to create and visualize graph positional encodings.

    Args:
        cfg: Hydra configuration object
    """
    print("=== Boolean Circuit Positional Encoding Visualization ===")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed
    key = jax.random.PRNGKey(cfg.test_seed)

    # Create output directory
    output_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Generate circuit layer sizes
    input_n = cfg.circuit.input_bits
    output_n = cfg.circuit.output_bits
    arity = cfg.circuit.arity

    if cfg.circuit.layer_sizes is None:
        layer_sizes = generate_layer_sizes(
            input_n, output_n, arity, layer_n=cfg.circuit.num_layers
        )
    else:
        layer_sizes = cfg.circuit.layer_sizes

    print(f"\nCircuit configuration:")
    print(f"  Input bits: {input_n}")
    print(f"  Output bits: {output_n}")
    print(f"  Arity: {arity}")
    print(f"  Layer sizes: {layer_sizes}")

    # Generate circuit
    wires, logits = gen_circuit(key, layer_sizes, arity=arity)
    print(f"  Generated {len(logits)} gate layers")

    # Build graph with positional encodings
    hidden_dim = 64  # Use a reasonable default
    graph = build_graph(
        wires=wires,
        logits=logits,
        input_n=input_n,
        arity=arity,
        hidden_dim=hidden_dim,
        bidirectional_edges=True,
    )

    # Print graph information
    print_graph_info(graph)

    # Plot positional encodings
    print(f"\n=== Creating Positional Encoding Visualizations ===")
    plot_positional_encodings(graph, output_dir)

    print(f"\n=== Visualization Complete ===")
    print(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    # Check if sklearn is available for PCA
    try:
        import sklearn
    except ImportError:
        print("Warning: scikit-learn not found. PCA visualization will be skipped.")
        print("Install with: pip install scikit-learn")

    main()

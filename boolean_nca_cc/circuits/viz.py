import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections  # Add this import for PatchCollection

from boolean_nca_cc.circuits.model import run_circuit


def visualize_network(wires, layer_sizes, title="Network Architecture"):
    """
    Visualize the wiring of a boolean circuit network.

    Args:
        wires: List of wire connection patterns for each layer
        layer_sizes: List of tuples (nodes, group_size) for each layer
        title: Title for the visualization

    Returns:
        None (displays the plot)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract number of nodes per layer
    nodes_per_layer = [size[0] for size in layer_sizes]
    total_layers = len(nodes_per_layer)

    # Determine y positions for each layer
    layer_y_positions = {}
    for layer_idx, node_count in enumerate(nodes_per_layer):
        # Create evenly spaced nodes
        spacing = 1.0 / (node_count + 1)
        positions = np.arange(spacing, 1.0, spacing)[:node_count]
        layer_y_positions[layer_idx] = positions

    # Draw nodes for each layer
    node_radius = 0.01 + 0.03 / np.log(max(nodes_per_layer))  # Adaptive node size
    colors = plt.cm.viridis(np.linspace(0, 1, total_layers))

    for layer_idx, node_count in enumerate(nodes_per_layer):
        # X position for this layer
        layer_x = layer_idx / (total_layers - 1) if total_layers > 1 else 0.5

        # Draw all nodes in this layer
        y_positions = layer_y_positions[layer_idx]
        circles = [
            plt.Circle((layer_x, y), node_radius, color=colors[layer_idx], alpha=0.7)
            for y in y_positions
        ]

        # Add collection of circles to the plot
        node_collection = collections.PatchCollection(circles, match_original=True)
        ax.add_collection(node_collection)

        # Add layer label
        ax.text(
            layer_x,
            1.05,
            f"Layer {layer_idx}\n({node_count} nodes)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

        # Add group indicators if not the input layer
        if layer_idx > 0:
            group_size = layer_sizes[layer_idx][1]
            arity = wires[layer_idx - 1].shape[0]

            # Highlight groups with background
            if group_size > 1:
                for g in range(node_count // group_size):
                    group_y_min = y_positions[g * group_size] - 1.5 * node_radius
                    group_y_max = (
                        y_positions[min((g + 1) * group_size - 1, node_count - 1)]
                        + 1.5 * node_radius
                    )
                    group_height = group_y_max - group_y_min
                    group_rect = plt.Rectangle(
                        (layer_x - 2 * node_radius, group_y_min),
                        4 * node_radius,
                        group_height,
                        color=colors[layer_idx],
                        alpha=0.2,
                    )
                    ax.add_patch(group_rect)

    # Draw connections between layers
    for layer_idx, layer_wires in enumerate(wires):
        source_layer = layer_idx
        target_layer = layer_idx + 1

        source_positions = layer_y_positions[source_layer]
        target_positions = layer_y_positions[target_layer]

        source_x = source_layer / (total_layers - 1) if total_layers > 1 else 0.5
        target_x = target_layer / (total_layers - 1) if total_layers > 1 else 0.5

        # Extract wire information
        arity = layer_wires.shape[0]
        group_size = layer_sizes[target_layer][1]

        # Draw connections for each target gate/group
        for group_idx in range(layer_wires.shape[1]):
            # Get target positions for this group
            group_targets = []
            for i in range(group_size):
                node_idx = group_idx * group_size + i
                if node_idx < len(target_positions):
                    group_targets.append(target_positions[node_idx])

            # Get source indices for this group
            for input_idx in range(arity):
                source_idx = layer_wires[input_idx, group_idx]
                if source_idx < len(source_positions):
                    # Draw a line from source to each target in the group
                    for target_y in group_targets:
                        # Use alpha to reduce visual clutter
                        alpha = 0.3 if len(group_targets) > 1 else 0.6
                        ax.plot(
                            [source_x, target_x],
                            [source_positions[source_idx], target_y],
                            "-",
                            color=colors[target_layer],
                            alpha=alpha,
                            linewidth=0.5,
                        )

    # Adjust plot appearance
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.1)
    ax.axis("off")
    ax.set_title(f"{title}\nArity: {arity}, Group Size: {group_size}", fontsize=14)

    plt.tight_layout()
    plt.show()

    return fig


def evaluate_and_visualize(
    logits, wires, x, y0, title_prefix="", hard=True, visualize=True
):
    """
    Evaluate the circuit, calculate accuracy, and visualize the results.

    Args:
        logits: List of logits for each layer (trained parameters).
        wires: List of wire connection patterns for the circuit.
        x: Input data tensor.
        y0: Target output data tensor.
        title_prefix: Optional string to prepend to the plot title.
        hard: Whether to run the circuit in hard mode (True) or soft mode (False).
        visualize: Whether to generate and display visualization plots.

    Returns:
        Dictionary containing evaluation metrics and error information.
    """
    # 1. Run the circuit in hard mode to get binary predictions
    acts = run_circuit(logits, wires, x, hard=hard)
    y_pred = acts[-1]

    # 2. Calculate accuracy
    y_pred_rounded = jp.round(y_pred) if not hard else y_pred
    correct_predictions = jp.equal(y_pred_rounded, y0)
    accuracy = jp.mean(correct_predictions)

    # 3. Calculate the difference map (errors)
    errors = jp.not_equal(y_pred_rounded, y0).astype(jp.float32)

    # 4. Find indices of errors
    error_indices = jp.where(errors)
    batch_indices = error_indices[0]
    bit_indices = error_indices[1]

    # 5. Create metrics dictionary
    metrics = {
        "accuracy": float(accuracy),
        "error_count": int(jp.sum(errors)),
        "total_bits": int(errors.size),
        "error_rate": float(jp.sum(errors) / errors.size),
        "error_locations": {
            "batch_indices": np.array(batch_indices),
            "bit_indices": np.array(bit_indices),
        },
    }

    # 6. For each error, record input and predicted/expected values
    if len(batch_indices) > 0:
        input_values = []
        for idx in np.array(batch_indices):
            input_values.append(np.array(x[idx]))

        metrics["error_details"] = {
            "input_values": input_values,
            "predicted_values": np.array(
                [y_pred_rounded[i, j] for i, j in zip(batch_indices, bit_indices)]
            ),
            "expected_values": np.array(
                [y0[i, j] for i, j in zip(batch_indices, bit_indices)]
            ),
        }

    # 7. Visualization if requested
    if visualize:
        # Convert to NumPy for plotting
        y_pred_np = np.array(y_pred_rounded)
        y0_np = np.array(y0)
        errors_np = np.array(errors)

        fig, axes = plt.subplots(3, 1, figsize=(20, 6), constrained_layout=True)
        cmap = "viridis"

        # Plot Predicted Output
        axes[0].imshow(
            y_pred_np.T,
            cmap=cmap,
            interpolation="nearest",
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        axes[0].set_title("Predicted Output")
        axes[0].set_xlabel("Batch Index")
        axes[0].set_ylabel("Output Bit Index")

        # Plot Target Output
        axes[1].imshow(
            y0_np.T, cmap=cmap, interpolation="nearest", vmin=0, vmax=1, aspect="auto"
        )
        axes[1].set_title("Target Output")
        axes[1].set_xlabel("Batch Index")
        axes[1].set_ylabel("Output Bit Index")

        # Plot Errors (highlighting incorrect bits)
        cmap_errors = plt.cm.colors.ListedColormap(["lightgray", "red"])
        axes[2].imshow(
            errors_np.T,
            cmap=cmap_errors,
            interpolation="nearest",
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        axes[2].set_title("Errors (Incorrect Bits)")
        axes[2].set_xlabel("Batch Index")
        axes[2].set_ylabel("Output Bit Index")

        # Add overall title with accuracy
        fig.suptitle(
            f"{title_prefix}Evaluation - Accuracy: {accuracy:.4f}", fontsize=16
        )
        plt.show()
        
    plt.savefig("nca_eval.png")

    return metrics

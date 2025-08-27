import jax
import jax.numpy as jp
import matplotlib.collections as collections  # Add this import for PatchCollection
import matplotlib.pyplot as plt
import numpy as np

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
    logits,
    wires,
    x,
    y0,
    title_prefix="",
    hard=True,
    visualize=True,
    train_test_split=False,
    test_ratio=0.2,
    split_seed=None,
):
    """
    Evaluate the circuit, calculate accuracy, and visualize the results.
    Optionally split data into train/test sets for memorization vs generalization analysis.

    Args:
        logits: List of logits for each layer (trained parameters).
        wires: List of wire connection patterns for the circuit.
        x: Input data tensor.
        y0: Target output data tensor.
        title_prefix: Optional string to prepend to the plot title.
        hard: Whether to run the circuit in hard mode (True) or soft mode (False).
        visualize: Whether to generate and display visualization plots.
        train_test_split: If True, split data into train/test and show separate visualizations.
        test_ratio: Fraction of data to use for testing (default 0.2 for 80/20 split).
        split_seed: Random seed for reproducible splitting (required if train_test_split=True).

    Returns:
        If train_test_split=False: Dictionary containing evaluation metrics
        If train_test_split=True: Tuple of (train_metrics, test_metrics) dictionaries
    """
    if not train_test_split:
        # Traditional mode: evaluate all data together
        return _evaluate_single_dataset(logits, wires, x, y0, title_prefix, hard, visualize)

    else:
        # Train/test split mode: split data and evaluate separately
        if split_seed is None:
            raise ValueError("split_seed is required when train_test_split=True")

        # Perform train/test split using the same logic as get_task_data
        case_n = x.shape[0]
        key = jax.random.PRNGKey(split_seed)

        # Generate random permutation indices
        indices = jax.random.permutation(key, case_n)

        # Calculate split point
        test_size = int(case_n * test_ratio)
        train_size = case_n - test_size

        # Split indices
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        # Split data
        x_train = x[train_indices]
        y_train = y0[train_indices]
        x_test = x[test_indices]
        y_test = y0[test_indices]

        # Evaluate both sets
        train_metrics = _evaluate_single_dataset(
            logits, wires, x_train, y_train, f"{title_prefix}Training Set - ", hard, visualize
        )

        test_metrics = _evaluate_single_dataset(
            logits, wires, x_test, y_test, f"{title_prefix}Test Set - ", hard, visualize
        )

        # Create combined visualization if requested
        if visualize:
            _create_train_test_comparison_plot(
                logits,
                wires,
                x_train,
                y_train,
                x_test,
                y_test,
                train_metrics,
                test_metrics,
                title_prefix,
                hard,
            )

        return train_metrics, test_metrics


def _evaluate_single_dataset(logits, wires, x, y0, title_prefix="", hard=True, visualize=True):
    """
    Helper function to evaluate a single dataset (used for both traditional and split modes).
    """
    # 1. Run the circuit to get predictions
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
                [y_pred_rounded[i, j] for i, j in zip(batch_indices, bit_indices, strict=False)]
            ),
            "expected_values": np.array(
                [y0[i, j] for i, j in zip(batch_indices, bit_indices, strict=False)]
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
        axes[1].imshow(y0_np.T, cmap=cmap, interpolation="nearest", vmin=0, vmax=1, aspect="auto")
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
        fig.suptitle(f"{title_prefix}Evaluation - Accuracy: {accuracy:.4f}", fontsize=16)
        plt.show()

    return metrics


def _create_train_test_comparison_plot(
    logits,
    wires,
    x_train,
    y_train,
    x_test,
    y_test,
    train_metrics,
    test_metrics,
    title_prefix="",
    hard=True,
):
    """
    Create a side-by-side comparison visualization of training and test performance.
    """
    # Get predictions for both sets
    train_acts = run_circuit(logits, wires, x_train, hard=hard)
    test_acts = run_circuit(logits, wires, x_test, hard=hard)

    train_pred = jp.round(train_acts[-1]) if not hard else train_acts[-1]
    test_pred = jp.round(test_acts[-1]) if not hard else test_acts[-1]

    # Calculate errors
    train_errors = jp.not_equal(train_pred, y_train).astype(jp.float32)
    test_errors = jp.not_equal(test_pred, y_test).astype(jp.float32)

    # Convert to numpy
    train_pred_np = np.array(train_pred)
    test_pred_np = np.array(test_pred)
    train_target_np = np.array(y_train)
    test_target_np = np.array(y_test)
    train_errors_np = np.array(train_errors)
    test_errors_np = np.array(test_errors)

    # Create comprehensive comparison plot
    fig, axes = plt.subplots(3, 2, figsize=(24, 10), constrained_layout=True)
    cmap = "viridis"
    cmap_errors = plt.cm.colors.ListedColormap(["lightgray", "red"])

    # Training set visualizations (left column)
    axes[0, 0].imshow(
        train_pred_np.T, cmap=cmap, interpolation="nearest", vmin=0, vmax=1, aspect="auto"
    )
    axes[0, 0].set_title(f"Training Predictions (Acc: {train_metrics['accuracy']:.4f})")
    axes[0, 0].set_xlabel("Sample Index")
    axes[0, 0].set_ylabel("Output Bit")

    axes[1, 0].imshow(
        train_target_np.T, cmap=cmap, interpolation="nearest", vmin=0, vmax=1, aspect="auto"
    )
    axes[1, 0].set_title("Training Targets")
    axes[1, 0].set_xlabel("Sample Index")
    axes[1, 0].set_ylabel("Output Bit")

    axes[2, 0].imshow(
        train_errors_np.T, cmap=cmap_errors, interpolation="nearest", vmin=0, vmax=1, aspect="auto"
    )
    axes[2, 0].set_title(f"Training Errors ({train_metrics['error_count']} total)")
    axes[2, 0].set_xlabel("Sample Index")
    axes[2, 0].set_ylabel("Output Bit")

    # Test set visualizations (right column)
    axes[0, 1].imshow(
        test_pred_np.T, cmap=cmap, interpolation="nearest", vmin=0, vmax=1, aspect="auto"
    )
    axes[0, 1].set_title(f"Test Predictions (Acc: {test_metrics['accuracy']:.4f})")
    axes[0, 1].set_xlabel("Sample Index")
    axes[0, 1].set_ylabel("Output Bit")

    axes[1, 1].imshow(
        test_target_np.T, cmap=cmap, interpolation="nearest", vmin=0, vmax=1, aspect="auto"
    )
    axes[1, 1].set_title("Test Targets")
    axes[1, 1].set_xlabel("Sample Index")
    axes[1, 1].set_ylabel("Output Bit")

    axes[2, 1].imshow(
        test_errors_np.T, cmap=cmap_errors, interpolation="nearest", vmin=0, vmax=1, aspect="auto"
    )
    axes[2, 1].set_title(f"Test Errors ({test_metrics['error_count']} total)")
    axes[2, 1].set_xlabel("Sample Index")
    axes[2, 1].set_ylabel("Output Bit")

    # Add overall title with generalization analysis
    acc_diff = train_metrics["accuracy"] - test_metrics["accuracy"]
    if acc_diff > 0.1:
        generalization_status = "Likely Memorizing"
        status_color = "red"
    elif acc_diff > 0.05:
        generalization_status = "Possible Overfitting"
        status_color = "orange"
    else:
        generalization_status = "Good Generalization"
        status_color = "green"

    fig.suptitle(
        f"{title_prefix}Train vs Test Comparison - {generalization_status}\n"
        f"Train Acc: {train_metrics['accuracy']:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | "
        f"Diff: {acc_diff:.4f}",
        fontsize=16,
        color=status_color,
    )

    plt.show()

    return fig


def create_wandb_visualization(logits, wires, x, y0, title_prefix="", hard=True):
    """
    Create visualization images for wandb logging without displaying them.

    Args:
        logits: List of logits for each layer (trained parameters).
        wires: List of wire connection patterns for the circuit.
        x: Input data tensor.
        y0: Target output data tensor.
        title_prefix: Optional string to prepend to the plot title.
        hard: Whether to run the circuit in hard mode (True) or soft mode (False).

    Returns:
        Dictionary containing matplotlib figures for wandb logging.
    """
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt

    # 1. Run the circuit to get predictions
    acts = run_circuit(logits, wires, x, hard=hard)
    y_pred = acts[-1]

    # 2. Calculate accuracy and errors
    y_pred_rounded = jp.round(y_pred) if not hard else y_pred
    correct_predictions = jp.equal(y_pred_rounded, y0)
    accuracy = jp.mean(correct_predictions)
    errors = jp.not_equal(y_pred_rounded, y0).astype(jp.float32)

    # 3. Convert to NumPy for plotting
    y_pred_np = np.array(y_pred_rounded)
    y0_np = np.array(y0)
    errors_np = np.array(errors)
    x_np = np.array(x)

    # 4. Create visualization figure
    fig, axes = plt.subplots(4, 1, figsize=(20, 8), constrained_layout=True)
    cmap = "viridis"

    # Plot Input
    axes[0].imshow(
        x_np.T,
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    axes[0].set_title("Input")
    axes[0].set_xlabel("Batch Index")
    axes[0].set_ylabel("Input Bit Index")

    # Plot Predicted Output
    axes[1].imshow(
        y_pred_np.T,
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    axes[1].set_title("Predicted Output")
    axes[1].set_xlabel("Batch Index")
    axes[1].set_ylabel("Output Bit Index")

    # Plot Target Output
    axes[2].imshow(y0_np.T, cmap=cmap, interpolation="nearest", vmin=0, vmax=1, aspect="auto")
    axes[2].set_title("Target Output")
    axes[2].set_xlabel("Batch Index")
    axes[2].set_ylabel("Output Bit Index")

    # Plot Errors (highlighting incorrect bits)
    cmap_errors = plt.cm.colors.ListedColormap(["lightgray", "red"])
    axes[3].imshow(
        errors_np.T,
        cmap=cmap_errors,
        interpolation="nearest",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    axes[3].set_title("Errors (Incorrect Bits)")
    axes[3].set_xlabel("Batch Index")
    axes[3].set_ylabel("Output Bit Index")

    # Add overall title with accuracy
    fig.suptitle(f"{title_prefix}Circuit Evaluation - Accuracy: {accuracy:.4f}", fontsize=16)

    return {
        "figure": fig,
        "accuracy": float(accuracy),
        "error_count": int(jp.sum(errors)),
        "total_bits": int(errors.size),
    }

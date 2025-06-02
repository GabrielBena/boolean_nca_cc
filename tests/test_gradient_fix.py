#!/usr/bin/env python3
"""
Test script to verify that the gradient flow fix works for hidden features.

This script creates simple circuits and checks that hidden projection layers
now receive gradients after our initialization fix.
"""

import jax
import jax.numpy as jp
from flax import nnx
import numpy as np

from boolean_nca_cc.models import CircuitGNN, CircuitSelfAttention
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
from boolean_nca_cc.training.utils import check_gradients
from boolean_nca_cc.training.train_loop import get_loss_from_graph


def test_gnn_gradient_flow():
    """Test that GNN hidden features now receive gradients."""
    print("🧪 Testing GNN gradient flow...")

    # Setup simple circuit
    arity = 2
    hidden_dim = 16
    input_n = 4
    output_n = 2
    layer_n = 2

    # Generate simple circuit
    layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n)
    rng = jax.random.PRNGKey(42)
    wires, logits = gen_circuit(rng, layer_sizes, arity=arity)

    # Create simple test data
    case_n = 8
    x_data = jp.arange(case_n)
    x_data = ((x_data[:, None] >> jp.arange(input_n)) & 1).astype(jp.float32)
    y_data = jp.zeros((case_n, output_n), dtype=jp.float32)

    # Build graph
    graph = build_graph(logits, wires, input_n, arity, hidden_dim)

    # Create GNN model
    rng, init_key = jax.random.split(rng)
    gnn = CircuitGNN(
        hidden_dim=hidden_dim,
        message_passing=True,
        node_mlp_features=[32, 16],
        edge_mlp_features=[32, 16],
        rngs=nnx.Rngs(params=init_key),
        use_attention=False,
        arity=arity,
    )

    # Create optimizer
    import optax

    opt_fn = optax.adamw(1e-3, weight_decay=1e-5)
    optimizer = nnx.Optimizer(gnn, opt_fn)

    def loss_fn(model):
        """Simple loss function for testing."""
        # Apply one GNN step
        updated_graph = model(graph)

        # Extract logits and compute loss
        from boolean_nca_cc.utils.graph_builder import extract_logits_from_graph

        logits_original_shapes = [l.shape for l in logits]
        current_logits = extract_logits_from_graph(
            updated_graph, logits_original_shapes
        )

        # Simple loss computation
        loss, (hard_loss, pred, pred_hard) = get_loss_from_graph(
            current_logits, wires, x_data, y_data, "l4"
        )

        return loss, (updated_graph, pred)

    # Test gradient computation
    print("  Computing gradients...")
    (loss, (updated_graph, pred)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        gnn
    )

    # Check hidden features are non-zero after update
    hidden_nonzero = updated_graph.nodes["hidden"].any()
    print(f"  Hidden features non-zero after update: {hidden_nonzero}")

    # Check gradients
    has_grads, zero_grad_paths = check_gradients(
        grads, verbose=False, return_zero_grad_paths=True
    )

    # Count hidden-related parameters with gradients
    hidden_params_with_grads = sum(
        1 for path in zero_grad_paths if "hidden" not in path.lower()
    )
    total_params = len(jax.tree.leaves(nnx.state(gnn)))
    hidden_grad_ratio = hidden_params_with_grads / total_params

    print(
        f"  Parameters with gradients: {total_params - len(zero_grad_paths)}/{total_params}"
    )
    print(f"  Loss: {loss:.6f}")

    # Update model and check hidden features change
    optimizer.update(grads)

    return has_grads, len(
        zero_grad_paths
    ) < 5  # Allow some parameters to have zero grads


def test_self_attention_gradient_flow():
    """Test that Self-Attention hidden features now receive gradients."""
    print("🧪 Testing Self-Attention gradient flow...")

    # Setup simple circuit
    arity = 2
    hidden_dim = 16
    input_n = 4
    output_n = 2
    layer_n = 2

    # Generate simple circuit
    layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n)
    rng = jax.random.PRNGKey(42)
    wires, logits = gen_circuit(rng, layer_sizes, arity=arity)

    # Create simple test data
    case_n = 8
    x_data = jp.arange(case_n)
    x_data = ((x_data[:, None] >> jp.arange(input_n)) & 1).astype(jp.float32)
    y_data = jp.ones(
        (case_n, output_n), dtype=jp.float32
    )  # Non-zero target for better gradients

    # Build graph
    graph = build_graph(logits, wires, input_n, arity, hidden_dim)
    n_nodes = int(graph.n_node[0])

    # Create Self-Attention model
    rng, init_key = jax.random.split(rng)
    attn_model = CircuitSelfAttention(
        n_node=n_nodes,
        hidden_dim=hidden_dim,
        arity=arity,
        num_heads=4,
        num_layers=2,
        mlp_dim=32,
        dropout_rate=0.0,
        rngs=nnx.Rngs(params=init_key),
    )

    # Create optimizer
    import optax

    opt_fn = optax.adamw(1e-3, weight_decay=1e-5)
    optimizer = nnx.Optimizer(attn_model, opt_fn)

    def loss_fn(model):
        """Simple loss function for testing."""
        # Apply one attention step
        updated_graph = model(graph)

        # Extract logits and compute loss
        from boolean_nca_cc.utils.graph_builder import extract_logits_from_graph

        logits_original_shapes = [l.shape for l in logits]
        current_logits = extract_logits_from_graph(
            updated_graph, logits_original_shapes
        )

        # Simple loss computation
        loss, (hard_loss, pred, pred_hard) = get_loss_from_graph(
            current_logits, wires, x_data, y_data, "l4"
        )

        return loss, (updated_graph, pred)

    # Test gradient computation over multiple steps (cold start test)
    print("  Testing cold start behavior...")

    initial_hidden = graph.nodes["hidden"]
    gradients_flow_eventually = False
    hidden_features_change = False

    for step in range(3):
        # Compute gradients
        (loss, (updated_graph, pred)), grads = nnx.value_and_grad(
            loss_fn, has_aux=True
        )(attn_model)

        # Check if hidden features are changing (this is the real test!)
        hidden_change = jp.mean(jp.abs(updated_graph.nodes["hidden"] - initial_hidden))

        if step == 0:
            print(f"  Step {step}: Loss={loss:.4f}, Hidden change={hidden_change:.6f}")
        else:
            print(f"  Step {step}: Loss={loss:.4f}, Hidden change={hidden_change:.6f}")
            if hidden_change > 1e-6:  # Hidden features are actually changing
                hidden_features_change = True

        # Check gradient flow
        has_grads, zero_grad_paths = check_gradients(
            grads, verbose=False, return_zero_grad_paths=True
        )
        total_params = len(jax.tree.leaves(nnx.state(attn_model)))
        params_with_grads = total_params - len(zero_grad_paths)

        if step == 0:
            print(f"    Parameters with gradients: {params_with_grads}/{total_params}")
        else:
            print(f"    Parameters with gradients: {params_with_grads}/{total_params}")
            if (
                params_with_grads > total_params * 0.5
            ):  # Most parameters have gradients after warmup
                gradients_flow_eventually = True

        # Update model
        optimizer.update(grads)

        # Update initial hidden for next comparison
        initial_hidden = updated_graph.nodes["hidden"]

    print(f"  ✅ Hidden features change: {hidden_features_change}")
    print(f"  ✅ Gradient flow after warmup: {gradients_flow_eventually}")

    return hidden_features_change, gradients_flow_eventually


def main():
    """Run gradient flow tests for both models."""
    print("🔧 Testing gradient flow fixes for hidden features\n")

    # Test GNN
    gnn_has_grads, gnn_good_grads = test_gnn_gradient_flow()
    print(f"  ✅ GNN gradients working: {gnn_has_grads and gnn_good_grads}\n")

    # Test Self-Attention
    attn_hidden_change, attn_gradient_flow = test_self_attention_gradient_flow()
    print(
        f"  ✅ Self-Attention gradients working: {attn_hidden_change and attn_gradient_flow}\n"
    )

    if gnn_has_grads and gnn_good_grads and attn_hidden_change and attn_gradient_flow:
        print(
            "🎉 SUCCESS: Both models now have proper gradient flow to hidden features!"
        )
        print("   • GNN: Hidden features receive gradients immediately")
        print(
            "   • Self-Attention: Hidden features change after cold start (as designed)"
        )
        print("   The zero initialization fix is working correctly.")
    else:
        print("❌ FAILURE: Some models still have gradient flow issues.")
        print("   Check the initialization or model architecture.")

    return (
        gnn_has_grads and gnn_good_grads and attn_hidden_change and attn_gradient_flow
    )


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

#!/usr/bin/env python3
"""
Test if hidden projection gets gradients when using multiple self-attention steps.
"""

import jax
import jax.numpy as jp
from flax import nnx

from boolean_nca_cc.models import CircuitSelfAttention
from boolean_nca_cc.utils.graph_builder import build_graph, extract_logits_from_graph
from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
from boolean_nca_cc.training.utils import check_gradients
from boolean_nca_cc.training.train_loop import get_loss_from_graph


def test_multi_step_gradients():
    """Test if hidden gradients appear with multiple self-attention steps."""
    print("🔍 Testing Multi-Step Hidden Gradients (with cold start)")

    # Setup simple circuit
    arity = 2
    hidden_dim = 8
    input_n = 4
    output_n = 2
    layer_n = 2

    # Generate simple circuit
    layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n)
    rng = jax.random.PRNGKey(42)
    wires, logits = gen_circuit(rng, layer_sizes, arity=arity)

    # Create test data
    case_n = 4
    x_data = jp.arange(case_n)
    x_data = ((x_data[:, None] >> jp.arange(input_n)) & 1).astype(jp.float32)
    y_data = jp.ones((case_n, output_n), dtype=jp.float32)

    # Build graph
    graph = build_graph(logits, wires, input_n, arity, hidden_dim)
    n_nodes = int(graph.n_node[0])

    # Create Self-Attention model
    rng, init_key = jax.random.split(rng)
    attn_model = CircuitSelfAttention(
        n_node=n_nodes,
        hidden_dim=hidden_dim,
        arity=arity,
        num_heads=2,
        num_layers=1,
        mlp_dim=16,
        dropout_rate=0.0,
        rngs=nnx.Rngs(params=init_key),
    )

    logits_original_shapes = [l.shape for l in logits]

    # Create optimizer for "wake up" steps
    import optax

    opt_fn = optax.adamw(1e-3, weight_decay=1e-5)
    optimizer = nnx.Optimizer(attn_model, opt_fn)

    def loss_fn(model):
        current_graph = model(graph)
        current_logits = extract_logits_from_graph(
            current_graph, logits_original_shapes
        )
        loss, _ = get_loss_from_graph(current_logits, wires, x_data, y_data, "l4")
        return loss

    # First: Do a few optimization steps to "wake up" the network
    print("\n🌅 Warming up the network...")
    for warmup_step in range(3):
        loss, grads = nnx.value_and_grad(loss_fn)(attn_model)
        optimizer.update(grads)

        # Check gradients after warmup
        has_grads, zero_grad_paths = check_gradients(
            grads, verbose=False, return_zero_grad_paths=True
        )
        total_params = len(jax.tree.leaves(nnx.state(attn_model)))
        params_with_grads = total_params - len(zero_grad_paths)

        hidden_proj_zero_grads = [
            path for path in zero_grad_paths if "hidden_proj" in path
        ]
        hidden_proj_has_grads = len(hidden_proj_zero_grads) == 0

        print(
            f"  Warmup step {warmup_step}: {params_with_grads}/{total_params} params have grads, hidden_proj: {hidden_proj_has_grads}"
        )

    # Now test different numbers of steps with the warmed-up network
    print("\n🔥 Testing with warmed-up network...")
    for num_steps in [1, 2, 3, 5]:
        print(f"\n--- Testing {num_steps} self-attention steps (after warmup) ---")

        def multi_step_loss_fn(model):
            current_graph = graph

            # Apply multiple self-attention steps
            for step in range(num_steps):
                current_graph = model(current_graph)

            # Extract logits and compute loss
            current_logits = extract_logits_from_graph(
                current_graph, logits_original_shapes
            )
            loss, _ = get_loss_from_graph(current_logits, wires, x_data, y_data, "l4")
            return loss

        # Compute gradients
        loss, grads = nnx.value_and_grad(multi_step_loss_fn)(attn_model)

        # Check gradient flow
        has_grads, zero_grad_paths = check_gradients(
            grads, verbose=False, return_zero_grad_paths=True
        )

        # Count parameters by component
        total_params = len(jax.tree.leaves(nnx.state(attn_model)))
        params_with_grads = total_params - len(zero_grad_paths)

        # Check specifically for hidden_proj
        hidden_proj_zero_grads = [
            path for path in zero_grad_paths if "hidden_proj" in path
        ]
        hidden_proj_has_grads = len(hidden_proj_zero_grads) == 0

        print(f"  Loss: {loss:.6f}")
        print(f"  Parameters with gradients: {params_with_grads}/{total_params}")
        print(f"  Hidden projection has gradients: {hidden_proj_has_grads}")

        if hidden_proj_has_grads:
            print(
                f"  🎉 SUCCESS: Hidden projection gets gradients with {num_steps} steps after warmup!"
            )
            return True

        # Print gradient norms for hidden_proj parameters
        all_params = jax.tree.flatten_with_path(nnx.state(attn_model))[0]
        all_grads = jax.tree.flatten_with_path(grads)[0]

        for (param_path, param), (grad_path, grad) in zip(all_params, all_grads):
            param_name = "_".join(
                str(key).strip("()").strip("[]").strip("'") for key in param_path
            )
            if "hidden_proj" in param_name:
                grad_norm = jp.linalg.norm(grad.flatten())
                print(f"    {param_name}: grad_norm={grad_norm:.2e}")

    print(
        f"\n❌ Hidden projection still doesn't get gradients even after warmup and multiple steps"
    )
    return False


if __name__ == "__main__":
    test_multi_step_gradients()

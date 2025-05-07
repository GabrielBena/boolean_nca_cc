# boolean_nca_cc/circuits/simple_fixed_circuit_demo.py

import jax
import jax.numpy as jp
import numpy as np
import optax
from flax import nnx
from tqdm.auto import trange # Use trange for a simple loop progress bar
from functools import partial
from typing import List, Tuple, Dict

# --- Imports from boolean_nca_cc ---
# Circuits
from boolean_nca_cc.circuits.model import make_nops, run_layer
from boolean_nca_cc.circuits import tasks
from boolean_nca_cc.circuits.training import binary_cross_entropy, compute_accuracy

# GNN Models
from boolean_nca_cc.models.gnn import CircuitGNN, run_gnn_scan

# Utils
from boolean_nca_cc.utils.positional_encoding import get_positional_encoding
# We'll use a specialized extract_final_lut_logits below
import jraph

# UI (Optional, can be added later based on FixedCircuitGNNDemo structure)
from imgui_bundle import implot, imgui, immapp, hello_imgui

# --- Constants and Configuration ---
INPUT_N = 4
NUM_LUTS = 3
ARITY = 2
OUTPUT_N = 1
CASE_N = 1 << INPUT_N

GNN_HIDDEN_DIM = 32
GNN_MESSAGE_PASSING_STEPS = 5 # How many steps the GNN runs to update LUTs
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5 # Total training iterations

# Node ID conventions
PRIMARY_INPUT_NODE_IDS = list(range(INPUT_N)) # 0, 1, 2, 3
LUT_NODE_IDS = [INPUT_N + i for i in range(NUM_LUTS)] # 4, 5, 6
# Assume LUT_NODE_IDS[-1] (node 6) is the output LUT for the circuit

# --- Topology Definitions ---
def get_circuit_connectivity(topology_type: str) -> Tuple[jp.ndarray, jp.ndarray]:
    """Returns senders and receivers for a given topology."""
    if topology_type == "cascade":
        senders = jp.array([0, 1,  4, 2,  5, 3], dtype=jp.int32)
        receivers = jp.array([4, 4,  5, 5,  6, 6], dtype=jp.int32)
    elif topology_type == "tree":
        senders = jp.array([0, 1,  2, 3,  4, 5], dtype=jp.int32)
        receivers = jp.array([4, 4,  5, 5,  6, 6], dtype=jp.int32)
    else:
        raise ValueError(f"Unknown topology_type: {topology_type}")
    return senders, receivers

# --- Boolean Circuit Evaluation for Fixed Topologies ---
@partial(jax.jit, static_argnames=("topology_type_static", "hard"))
def evaluate_fixed_circuit(
    topology_type_static: str, # JIT requires strings to be static
    lut_logits_list: List[jp.ndarray], # List of 3 logit arrays, each (1,1, 2**ARITY)
    x_batch: jp.ndarray,
    hard: bool = False
) -> jp.ndarray:
    luts_probs = [jax.nn.sigmoid(logits) for logits in lut_logits_list]
    if hard:
        luts_probs = [jp.round(probs) for probs in luts_probs]
    lut_a_p, lut_b_p, lut_c_p = luts_probs[0], luts_probs[1], luts_probs[2]

    if topology_type_static == "cascade":
        out_a = run_layer(lut_a_p, [x_batch[:, 0:1], x_batch[:, 1:2]])
        out_b = run_layer(lut_b_p, [out_a,       x_batch[:, 2:3]])
        final_out = run_layer(lut_c_p, [out_b,     x_batch[:, 3:4]])
    elif topology_type_static == "tree":
        out_a = run_layer(lut_a_p, [x_batch[:, 0:1], x_batch[:, 1:2]])
        out_b = run_layer(lut_b_p, [x_batch[:, 2:3], x_batch[:, 3:4]])
        final_out = run_layer(lut_c_p, [out_a,       out_b])
    else:
        # JAX JIT Limitation: Can't raise ValueError with dynamic content in JIT.
        # This path should ideally not be hit if topology_type_static is validated before JIT.
        # For safety, return NaNs or zeros if an unknown type somehow gets through.
        final_out = jp.full_like(x_batch[:, 0:1], jp.nan)

    return final_out.squeeze(axis=-1)

# --- Graph Building and Logit Extraction ---
def build_fixed_topology_graph(
    topology_type: str,
    initial_lut_logits_list: List[jp.ndarray], # List of 3 logit arrays, each (1,1, 2**ARITY)
    gnn_hidden_dim: int,
    rng_key_for_hidden_init: jax.random.PRNGKey, # Added RNG key for hidden state initialization
    current_circuit_loss: float = 0.0
) -> jraph.GraphsTuple:
    num_total_nodes = INPUT_N + NUM_LUTS
    senders, receivers = get_circuit_connectivity(topology_type)
    node_features = {
        "layer_idx_pe": jp.zeros(num_total_nodes, dtype=jp.int32), # Refined PE below
        "gate_id": jp.arange(num_total_nodes, dtype=jp.int32),
        "logits": jp.zeros((num_total_nodes, 2**ARITY), dtype=jp.float32),
        # Initialize hidden states with small random noise
        "hidden": jax.random.normal(rng_key_for_hidden_init, (num_total_nodes, gnn_hidden_dim)) * 0.01,
        "layer_pe": jp.zeros((num_total_nodes, gnn_hidden_dim), dtype=jp.float32),
        "intra_layer_pe": jp.zeros((num_total_nodes, gnn_hidden_dim), dtype=jp.float32)
    }
    for i, lut_node_id in enumerate(LUT_NODE_IDS):
        node_features["logits"] = node_features["logits"].at[lut_node_id].set(
            initial_lut_logits_list[i].flatten() # Store flat logits
        )
    layer_indices_for_pe_np = np.zeros(num_total_nodes, dtype=np.int32)
    if topology_type == "cascade":
        layer_indices_for_pe_np[LUT_NODE_IDS[0]] = 1; layer_indices_for_pe_np[LUT_NODE_IDS[1]] = 2; layer_indices_for_pe_np[LUT_NODE_IDS[2]] = 3
        max_pe_layer = 3
    elif topology_type == "tree":
        layer_indices_for_pe_np[LUT_NODE_IDS[0]] = 1; layer_indices_for_pe_np[LUT_NODE_IDS[1]] = 1; layer_indices_for_pe_np[LUT_NODE_IDS[2]] = 2
        max_pe_layer = 2
    else: # Should not happen
        max_pe_layer = 1
    node_features["layer_idx_pe"] = jp.array(layer_indices_for_pe_np)
    node_features["layer_pe"] = get_positional_encoding(node_features["layer_idx_pe"], gnn_hidden_dim, max_val=max_pe_layer + 1)
    node_features["intra_layer_pe"] = get_positional_encoding(jp.arange(num_total_nodes), gnn_hidden_dim, max_val=num_total_nodes)
    node_features["layer"] = jp.array(layer_indices_for_pe_np)

    return jraph.GraphsTuple(
        nodes=node_features, edges=None, senders=senders, receivers=receivers,
        n_node=jp.array([num_total_nodes]), n_edge=jp.array([len(senders)]),
        globals=jp.array(current_circuit_loss, dtype=jp.float32)
    )

def extract_final_lut_logits(
    graph: jraph.GraphsTuple,
    original_logit_shape: Tuple = (1, 1, 2**ARITY)
) -> List[jp.ndarray]:
    updated_lut_logits_list = []
    all_node_logits_flat = graph.nodes['logits']
    for lut_node_id in LUT_NODE_IDS:
        flat_logits = all_node_logits_flat[lut_node_id]
        reshaped_logits = flat_logits.reshape(original_logit_shape)
        updated_lut_logits_list.append(reshaped_logits)
    return updated_lut_logits_list

# --- Training Step Definition (similar to meta_train_step's loss_fn in train_loop.py) ---

# We need to define the loss function that will be differentiated.
# This function will take the GNN's *parameters* (or the GNN model state for nnx)
# and the *static inputs* for one training instance.
def compute_loss_for_gnn_update(
    gnn_model_instance: CircuitGNN, # The GNN model instance with current parameters
    initial_lut_logits_list_jax: List[jp.ndarray],
    x_task_data_jax: jp.ndarray,
    y_task_target_jax: jp.ndarray,
    # Removed '*' marker, all args can be positional or keyword based on call
    static_rng_key_for_graph: jax.random.PRNGKey, 
    static_topology_type: str,
    static_gnn_steps: int,
    static_initial_lut_shape: Tuple # e.g. (1,1, 2**ARITY)
):
    # gnn_model_instance is the GNN with current parameters, and it's the one being differentiated.

    # 1. Build graph for the SPECIFIC topology
    # Estimate initial loss for graph globals (can be simplified or taken from previous step)
    temp_eval_output = evaluate_fixed_circuit(static_topology_type, initial_lut_logits_list_jax, x_task_data_jax, hard=False)
    # initial_circuit_loss_val = binary_cross_entropy(temp_eval_output, y_task_target_jax) # This is not used for current_circuit_loss now

    gnn_graph = build_fixed_topology_graph(
        static_topology_type, initial_lut_logits_list_jax,
        gnn_model_instance.hidden_dim,
        rng_key_for_hidden_init=static_rng_key_for_graph, # Pass the RNG key
        current_circuit_loss=0.0 # Pass 0.0 instead of initial_circuit_loss_val
    )

    # --- Debug: Print initial graph node features before GNN scan ---
    # This will only print when JAX traces the function, or if called outside JIT
    # For the first step, this part of compute_loss_for_gnn_update is traced.
    if 'print_debug_once' not in compute_loss_for_gnn_update.__dict__:
        print("\n--- DEBUG: Graph features BEFORE run_gnn_scan (first trace) ---")
        print(f"  Initial gnn_graph.nodes[\'hidden\'] (mean): {jp.mean(gnn_graph.nodes['hidden']):.6f}, (std): {jp.std(gnn_graph.nodes['hidden']):.6f}")
        print(f"  Initial gnn_graph.nodes[\'logits\'] (mean): {jp.mean(gnn_graph.nodes['logits']):.6f}, (std): {jp.std(gnn_graph.nodes['logits']):.6f}")
        compute_loss_for_gnn_update.print_debug_once = True # Ensure it prints only on first trace
    # --- End Debug ---

    # 2. Run GNN on this graph
    updated_gnn_graph = run_gnn_scan(gnn_model_instance, gnn_graph, static_gnn_steps)

    # --- Debug: Print updated graph node features AFTER GNN scan ---
    # Similar to above, this will print on first trace if we can manage the flag across JIT boundaries
    # For simplicity, we might rely on the print from the main loop for epoch 0 for after GNN scan info
    # Or, pass these as aux_data if we want to see them specifically from the first grad_fn call.
    # For now, let's focus on the gradient printout which is more direct.
    # --- End Debug ---

    # 3. Extract updated LUT logits for the 3 LUTs
    final_lut_logits_list = extract_final_lut_logits(updated_gnn_graph, static_initial_lut_shape)

    # 4. Evaluate the Boolean circuit
    circuit_output_soft = evaluate_fixed_circuit(static_topology_type, final_lut_logits_list, x_task_data_jax, hard=False)
    circuit_output_hard = evaluate_fixed_circuit(static_topology_type, final_lut_logits_list, x_task_data_jax, hard=True)

    # 5. Calculate task loss and accuracy
    task_loss = binary_cross_entropy(circuit_output_soft, y_task_target_jax)
    accuracy = compute_accuracy(circuit_output_hard, y_task_target_jax)

    return task_loss, (accuracy, circuit_output_hard, final_lut_logits_list)


# --- Main Application Logic ---
def main():
    print("Starting Fixed Circuit GNN Demo...")
    rng_key = jax.random.PRNGKey(43) # Main key for the script

    # --- Initialize Model and Optimizer ---
    gnn_init_key, rng_key = jax.random.split(rng_key)
    graph_build_key, rng_key = jax.random.split(rng_key) # Key for graph building's hidden init

    gnn_model = CircuitGNN(
        node_mlp_features=[GNN_HIDDEN_DIM]*2, edge_mlp_features=[GNN_HIDDEN_DIM]*2,
        hidden_dim=GNN_HIDDEN_DIM, arity=ARITY, message_passing=True, use_attention=False,
        rngs=nnx.Rngs(params=gnn_init_key)
    )
    optimizer = nnx.Optimizer(gnn_model, optax.adamw(learning_rate=LEARNING_RATE))

    # --- Initial LUTs (e.g., NOPs) ---
    # make_nops returns shape (gate_n // group_size, group_size, 2**arity)
    # For a single LUT, gate_n=1, group_size=1. So shape (1,1, 2**ARITY)
    single_lut_shape = (1, 1, 2**ARITY)
    initial_lut_logits_list = [make_nops(1, ARITY, 1) for _ in range(NUM_LUTS)]
    initial_lut_logits_list_jax = [jp.array(l) for l in initial_lut_logits_list] # Ensure JAX arrays

    # --- Task Data ---
    # TODO: Make these selectable via UI later
    current_topology = "cascade"
    target_task_name = "parity"

    x_task_data, y_task_target_original = tasks.get_task_data(
        target_task_name, case_n=CASE_N, input_bits=INPUT_N, output_bits=OUTPUT_N
    )
    if OUTPUT_N == 1: # Ensure y_task_target is (batch, 1)
        y_task_target = y_task_target_original[:, 0:1] if y_task_target_original.ndim == 2 and y_task_target_original.shape[1] > 1 else \
                        y_task_target_original[:, None] if y_task_target_original.ndim == 1 else y_task_target_original
    else:
        y_task_target = y_task_target_original[:, :OUTPUT_N]

    x_task_data_jax = jp.array(x_task_data)
    y_task_target_jax = jp.array(y_task_target)

    # --- Value and Grad Function ---
    # gnn_model is an nnx.Module, its state includes parameters.
    # value_and_grad will operate on the GNN module directly.
    grad_fn = nnx.value_and_grad(compute_loss_for_gnn_update, has_aux=True)

    # --- Initial Evaluation (Before Training) ---
    print("\n--- Initial State Evaluation ---")
    initial_loss, (initial_acc, _, initial_luts_from_eval) = compute_loss_for_gnn_update(
        gnn_model,  # Current gnn_model state (params are not yet trained by optimizer)
        initial_lut_logits_list_jax,
        x_task_data_jax,
        y_task_target_jax,
        # Pass static args directly, nnx.value_and_grad's JIT should handle them
        static_rng_key_for_graph=graph_build_key,
        static_topology_type=current_topology,
        static_gnn_steps=GNN_MESSAGE_PASSING_STEPS,
        static_initial_lut_shape=single_lut_shape
    )
    print(f"Initial Loss (from NOPs via GNN): {initial_loss:.4f}, Initial Accuracy: {initial_acc:.4f}")
    print("Initial LUTs (as provided to GNN):")
    for i, lut_logits_jax in enumerate(initial_lut_logits_list_jax):
        lut_probs = jax.nn.sigmoid(lut_logits_jax).flatten()
        print(f"  LUT {LUT_NODE_IDS[i]} (Node {i+INPUT_N}) initial probabilities (input to GNN):")
        for row_idx in range(2**ARITY):
            inputs_bin = format(row_idx, f'0{ARITY}b')
            print(f"    {inputs_bin} -> {lut_probs[row_idx]:.3f} (Hard: {jp.round(lut_probs[row_idx])})")

    print("LUTs after GNN processing initial NOPs (before any training):")
    for i, lut_logits_jax in enumerate(initial_luts_from_eval): # These are LUTs produced by GNN from NOPs
        lut_probs = jax.nn.sigmoid(lut_logits_jax).flatten()
        print(f"  LUT {LUT_NODE_IDS[i]} (Node {i+INPUT_N}) GNN-processed initial probabilities:")
        for row_idx in range(2**ARITY):
            inputs_bin = format(row_idx, f'0{ARITY}b')
            print(f"    {inputs_bin} -> {lut_probs[row_idx]:.3f} (Hard: {jp.round(lut_probs[row_idx])})")
    print("-----------------------------")

    # --- Training Loop ---
    print(f"\nTraining GNN for {current_topology} to learn {target_task_name}...")
    losses_log = []
    accuracies_log = []

    for epoch in trange(NUM_EPOCHS, desc="Training GNN"):
        # In nnx, the model instance `gnn_model` is stateful and updated by the optimizer.
        (loss_val, (acc_val, _, updated_luts_after_step)), grads = grad_fn( 
            gnn_model, # Pass the GNN model instance itself
            initial_lut_logits_list_jax, # GNN always starts from these initial NOPs
            x_task_data_jax,
            y_task_target_jax,
            # Pass static args directly, nnx.value_and_grad's JIT should handle them
            static_rng_key_for_graph=graph_build_key,
            static_topology_type=current_topology,
            static_gnn_steps=GNN_MESSAGE_PASSING_STEPS,
            static_initial_lut_shape=single_lut_shape
        )
        # Store parameters before optimizer update for comparison in the first epoch
        params_before_update = None
        if epoch == 0 and hasattr(gnn_model, 'node_update') and hasattr(gnn_model.node_update, 'mlp') and \
           hasattr(gnn_model.node_update.mlp, 'layers') and len(gnn_model.node_update.mlp.layers) > 0 and \
           hasattr(gnn_model.node_update.mlp.layers[0], 'kernel'):
            params_before_update = jp.copy(gnn_model.node_update.mlp.layers[0].kernel.value)

        optimizer.update(grads) # This updates gnn_model in-place

        if epoch == 0:
            print("\n--- After 1st Training Step ---")
            print(f"Step 0 Loss: {loss_val:.4f}, Step 0 Accuracy: {acc_val:.4f}")

            print("Gradient Statistics for all GNN parameters:")
            def print_grad_stats(path, leaf_grad):
                if hasattr(leaf_grad, 'shape') and hasattr(leaf_grad, 'dtype'): # Check if it's array-like
                    processed_leaf_grad = jp.asarray(leaf_grad) if not isinstance(leaf_grad, jp.ndarray) else leaf_grad
                    print(f"  Path: {path}") # Simplified path printing
                    print(f"    Mean: {jp.mean(processed_leaf_grad):.6f}, Std: {jp.std(processed_leaf_grad):.6f}, Min: {jp.min(processed_leaf_grad):.6f}, Max: {jp.max(processed_leaf_grad):.6f}")
                else:
                    print(f"  Path: {path}, Type: {type(leaf_grad)}, Value: {leaf_grad}") # Simplified path printing
            jax.tree_util.tree_map_with_path(print_grad_stats, grads)

            # Check if parameters of a specific layer changed
            if params_before_update is not None and hasattr(gnn_model, 'node_update') and hasattr(gnn_model.node_update, 'mlp') and \
               hasattr(gnn_model.node_update.mlp, 'layers') and len(gnn_model.node_update.mlp.layers) > 0 and \
               hasattr(gnn_model.node_update.mlp.layers[0], 'kernel'):
                params_after_update = gnn_model.node_update.mlp.layers[0].kernel.value
                print("Checking change in gnn_model.node_update.mlp.layers[0].kernel.value after optimizer update:")
                if jp.array_equal(params_before_update, params_after_update):
                    print("  Parameters DID NOT CHANGE.")
                else:
                    print("  Parameters DID CHANGE.")
                    print(f"    Original mean: {jp.mean(params_before_update):.6f}, New mean: {jp.mean(params_after_update):.6f}")
            else:
                print("Could not perform parameter change check for gnn_model.node_update.mlp.layers[0].kernel.value")

            print("LUTs after GNN update in 1st training step:")
            for i, lut_logits_jax in enumerate(updated_luts_after_step):
                lut_probs = jax.nn.sigmoid(lut_logits_jax).flatten()
                print(f"  LUT {LUT_NODE_IDS[i]} (Node {i+INPUT_N}) probabilities:")
                for row_idx in range(2**ARITY):
                    inputs_bin = format(row_idx, f'0{ARITY}b')
                    print(f"    {inputs_bin} -> {lut_probs[row_idx]:.3f} (Hard: {jp.round(lut_probs[row_idx])})")
            print("-----------------------------")

        losses_log.append(float(loss_val))
        accuracies_log.append(float(acc_val))

        if epoch % 100 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"Epoch {epoch}: Loss = {loss_val:.4f}, Accuracy = {acc_val:.4f}")

    print("Training finished.")

    # --- Final Evaluation (Example) ---
    print("\nEvaluating final GNN-produced LUTs:")
    # Get the final LUTs produced by the GNN after training
    # (This requires running the GNN forward pass one last time without gradient calculation)
    # For simplicity, we can use the aux output from the last training step,
    # or re-evaluate:
    _ , (final_acc, final_circ_hard, final_luts_jax) = compute_loss_for_gnn_update(
            gnn_model, initial_lut_logits_list_jax, x_task_data_jax, y_task_target_jax,
            # Pass static args directly, nnx.value_and_grad's JIT should handle them
            static_rng_key_for_graph=graph_build_key,
            static_topology_type=current_topology, static_gnn_steps=GNN_MESSAGE_PASSING_STEPS,
            static_initial_lut_shape=single_lut_shape
    )

    print(f"Final Accuracy with GNN-derived LUTs: {final_acc:.4f}")
    for i, lut_logits_jax in enumerate(final_luts_jax):
        lut_probs = jax.nn.sigmoid(lut_logits_jax).flatten()
        print(f"  LUT {LUT_NODE_IDS[i]} (Node {i+INPUT_N}) final probabilities:")
        for row_idx in range(2**ARITY):
            inputs_bin = format(row_idx, f'0{ARITY}b')
            print(f"    {inputs_bin} -> {lut_probs[row_idx]:.3f} (Hard: {jp.round(lut_probs[row_idx])})")

    # --- TODO: Add UI Here, similar to FixedCircuitGNNDemo class structure ---
    # The UI would call a method like `run_one_training_or_evaluation_step` from the
    # FixedCircuitGNNDemo class, which would wrap the call to grad_fn and optimizer.update.

if __name__ == "__main__":
    main()
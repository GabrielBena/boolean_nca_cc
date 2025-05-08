# boolean_nca_cc/circuits/fixed_circuits_gnn_demo.py

import jax
import jax.numpy as jp
import numpy as np
import optax
from flax import nnx # Using nnx as per the new training scripts
from tqdm.auto import tqdm
from functools import partial
from typing import List, Tuple, Dict, Union, Callable

# Imports from boolean_nca_cc
from boolean_nca_cc.circuits.model import make_nops, run_layer
from boolean_nca_cc.circuits import tasks
from boolean_nca_cc.circuits.training import binary_cross_entropy, compute_accuracy
from boolean_nca_cc.utils.graph_builder import get_positional_encoding # Direct import
from boolean_nca_cc.utils.extraction import extract_logits_from_graph as extract_logits_from_graph_general
from boolean_nca_cc.models.gnn import CircuitGNN, run_gnn_scan
import jraph

# For UI
from imgui_bundle import implot, imgui, immapp, hello_imgui
import PIL.Image
import PIL.ImageDraw

# --- Constants and Configuration ---
INPUT_N = 4
NUM_LUTS = 3
ARITY = 2  # Each LUT is 2-input
OUTPUT_N = 1 # Final circuit output is 1 bit
GNN_HIDDEN_DIM = 32
GNN_MESSAGE_PASSING_STEPS = 5 # Number of GNN steps to update LUTs
LEARNING_RATE = 1e-3
EPOCHS_PER_UI_UPDATE = 1 # For interactive training
CASE_N = 1 << INPUT_N

# Node IDs:
# Primary Inputs (PI): 0, 1, 2, 3
# LUTs: 4 (LUT_A), 5 (LUT_B), 6 (LUT_C)
PRIMARY_INPUT_NODE_IDS = list(range(INPUT_N))
LUT_NODE_IDS = [INPUT_N + i for i in range(NUM_LUTS)]
OUTPUT_LUT_NODE_ID = LUT_NODE_IDS[-1] # Assume the last LUT (LUT_C, node 6) is the output

# --- Topology Definitions ---
def get_circuit_connectivity(topology_type: str) -> Tuple[jp.ndarray, jp.ndarray, int]:
    """
    Returns senders, receivers, and the *local* index of the output LUT.
    """
    # LUTs are 0-indexed locally (0, 1, 2) mapped to global IDs 4, 5, 6
    # Global Node IDs: PIs = 0,1,2,3; LUTs = 4,5,6

    if topology_type == "cascade":  # Picture 1
        # LUT_A (node 4) inputs: PI_0 (node 0), PI_1 (node 1)
        # LUT_B (node 5) inputs: LUT_A_out (node 4), PI_2 (node 2)
        # LUT_C (node 6) inputs: LUT_B_out (node 5), PI_3 (node 3)
        senders = jp.array([0, 1,  4, 2,  5, 3], dtype=jp.int32)
        receivers = jp.array([4, 4,  5, 5,  6, 6], dtype=jp.int32)
        output_lut_local_idx = 2  # LUT_C is the 3rd LUT (index 2)
    elif topology_type == "tree":  # Picture 2
        # LUT_A (node 4) inputs: PI_0 (node 0), PI_1 (node 1)
        # LUT_B (node 5) inputs: PI_2 (node 2), PI_3 (node 3)
        # LUT_C (node 6) inputs: LUT_A_out (node 4), LUT_B_out (node 5)
        senders = jp.array([0, 1,  2, 3,  4, 5], dtype=jp.int32)
        receivers = jp.array([4, 4,  5, 5,  6, 6], dtype=jp.int32)
        output_lut_local_idx = 2  # LUT_C is the 3rd LUT (index 2)
    else:
        raise ValueError(f"Unknown topology_type: {topology_type}")
    return senders, receivers, output_lut_local_idx

# --- Boolean Circuit Evaluation for Fixed Topologies ---
@partial(jax.jit, static_argnames=("topology_type", "hard"))
def evaluate_fixed_circuit(
    topology_type: str,
    lut_logits_list: List[jp.ndarray], # List of 3 logit arrays, each shape (1,1, 2**ARITY)
    x_batch: jp.ndarray,
    hard: bool = False
) -> jp.ndarray:
    """
    Evaluates a fixed 4-input, 3-LUT, 1-output circuit.
    """
    luts_probs = [jax.nn.sigmoid(logits) for logits in lut_logits_list]
    if hard:
        luts_probs = [jp.round(probs) for probs in luts_probs]

    lut_a_probs, lut_b_probs, lut_c_probs = luts_probs[0], luts_probs[1], luts_probs[2]

    # x_batch[:, i:i+1] ensures the input to run_layer maintains a batch dimension
    if topology_type == "cascade":
        out_lut_a = run_layer(lut_a_probs, [x_batch[:, 0:1], x_batch[:, 1:2]])
        out_lut_b = run_layer(lut_b_probs, [out_lut_a,       x_batch[:, 2:3]])
        final_output = run_layer(lut_c_probs, [out_lut_b,     x_batch[:, 3:4]])
    elif topology_type == "tree":
        out_lut_a = run_layer(lut_a_probs, [x_batch[:, 0:1], x_batch[:, 1:2]])
        out_lut_b = run_layer(lut_b_probs, [x_batch[:, 2:3], x_batch[:, 3:4]])
        final_output = run_layer(lut_c_probs, [out_lut_a,       out_lut_b])
    else:
        raise ValueError(f"Unknown topology_type in evaluate_fixed_circuit: {topology_type}")

    return final_output.squeeze(axis=-1) # Squeeze the last dimension (group_size dim)

# --- Graph Building and Logit Extraction ---
def build_fixed_topology_graph(
    topology_type: str,
    initial_lut_logits_list: List[jp.ndarray], # List of 3 logit arrays, each (1,1, 2**ARITY)
    gnn_hidden_dim: int,
    current_circuit_loss: float = 0.0
) -> jraph.GraphsTuple:
    """Builds a jraph.GraphsTuple for the specified fixed circuit topology."""
    num_total_nodes = INPUT_N + NUM_LUTS
    senders, receivers, _ = get_circuit_connectivity(topology_type)

    # Initialize node features
    node_features = {
        "layer": jp.zeros(num_total_nodes, dtype=jp.int32), # Layer for PE
        "gate_id": jp.arange(num_total_nodes, dtype=jp.int32), # Global ID
        "logits": jp.zeros((num_total_nodes, 2**ARITY), dtype=jp.float32), # PI logits are 0
        "hidden": jp.zeros((num_total_nodes, gnn_hidden_dim), dtype=jp.float32),
        "layer_pe": jp.zeros((num_total_nodes, gnn_hidden_dim), dtype=jp.float32),
        "intra_layer_pe": jp.zeros((num_total_nodes, gnn_hidden_dim), dtype=jp.float32)
    }

    # Assign initial logits to LUT nodes (flattening them for storage in graph)
    for i, lut_node_id in enumerate(LUT_NODE_IDS):
        node_features["logits"] = node_features["logits"].at[lut_node_id].set(
            initial_lut_logits_list[i].flatten()
        )

    # Define layer indices for Positional Encoding
    # PIs: layer 0. LUTs directly connected to PIs: layer 1. Next LUTs: layer 2.
    layer_indices_for_pe = np.zeros(num_total_nodes, dtype=np.int32)
    if topology_type == "cascade":
        layer_indices_for_pe[LUT_NODE_IDS[0]] = 1 # LUT_A
        layer_indices_for_pe[LUT_NODE_IDS[1]] = 2 # LUT_B
        layer_indices_for_pe[LUT_NODE_IDS[2]] = 3 # LUT_C
        max_pe_layer = 3
    elif topology_type == "tree":
        layer_indices_for_pe[LUT_NODE_IDS[0]] = 1 # LUT_A
        layer_indices_for_pe[LUT_NODE_IDS[1]] = 1 # LUT_B
        layer_indices_for_pe[LUT_NODE_IDS[2]] = 2 # LUT_C
        max_pe_layer = 2
    node_features["layer_pe"] = get_positional_encoding(jp.array(layer_indices_for_pe), gnn_hidden_dim, max_val=max_pe_layer + 1)

    # Intra-layer PE (simple global index for now, could be refined)
    node_features["intra_layer_pe"] = get_positional_encoding(jp.arange(num_total_nodes), gnn_hidden_dim, max_val=num_total_nodes)


    return jraph.GraphsTuple(
        nodes=node_features,
        edges=None, # No edge features currently
        senders=senders,
        receivers=receivers,
        n_node=jp.array([num_total_nodes]),
        n_edge=jp.array([len(senders)]),
        globals=jp.array(current_circuit_loss, dtype=jp.float32)
    )

def extract_final_lut_logits(
    graph: jraph.GraphsTuple,
    original_logit_shape: Tuple = (1, 1, 2**ARITY) # Shape like from make_nops
) -> List[jp.ndarray]:
    """Extracts LUT logits from graph,reshaping them to original_logit_shape."""
    updated_lut_logits_list = []
    all_node_logits_flat = graph.nodes['logits'] # Shape (num_total_nodes, 2**ARITY)
    for lut_node_id in LUT_NODE_IDS:
        flat_logits = all_node_logits_flat[lut_node_id]
        reshaped_logits = flat_logits.reshape(original_logit_shape)
        updated_lut_logits_list.append(reshaped_logits)
    return updated_lut_logits_list


# --- Main Demo Class ---
class FixedCircuitGNNDemo:
    def __init__(self):
        self.rng_key = jax.random.key(42) # For nnx modules

        self.current_topology = "cascade"
        self.target_task_name = "parity" # Default task
        self.available_tasks = list(tasks.TASKS.keys())
        self.available_topologies = ["cascade", "tree"]

        self.gnn_model = CircuitGNN(
            node_mlp_features=[GNN_HIDDEN_DIM]*2,
            edge_mlp_features=[GNN_HIDDEN_DIM]*2,
            hidden_dim=GNN_HIDDEN_DIM,
            arity=ARITY,
            message_passing=True,
            use_attention=False,
            rngs=nnx.Rngs(params=jax.random.key(43)) # Use a new key
        )
        self.optimizer = nnx.Optimizer(self.gnn_model, optax.adamw(learning_rate=LEARNING_RATE))

        self.single_lut_shape = (1, 1, 2**ARITY) # Expected by run_layer
        self.initial_lut_logits_list = [make_nops(1, ARITY, 1) for _ in range(NUM_LUTS)]

        self._load_task_data()

        self.losses = []
        self.accuracies = []
        self.current_loss = 0.0
        self.current_accuracy = 0.0
        self.train_step_count = 0
        self.is_training = True

        # For UI display
        self.display_circuit_output_hard = np.zeros((CASE_N, OUTPUT_N), dtype=np.float32)
        self.display_lut_truth_tables_probs = [
            np.zeros(2**ARITY, dtype=np.float32) for _ in range(NUM_LUTS)
        ]
        # Run one step to initialize display values
        self.run_one_training_or_evaluation_step()

    def _load_task_data(self):
        self.x_task_data, self.y_task_target_original = tasks.get_task_data(
            self.target_task_name,
            case_n=CASE_N,
            input_bits=INPUT_N,
            output_bits=OUTPUT_N # Target task might define more outputs, we take the first
        )
        # Ensure y_task_target is (batch, 1) for binary_cross_entropy with single output
        if OUTPUT_N == 1:
            if self.y_task_target_original.ndim == 2 and self.y_task_target_original.shape[1] > 1:
                self.y_task_target = self.y_task_target_original[:, 0:1]
            elif self.y_task_target_original.ndim == 1:
                self.y_task_target = self.y_task_target_original[:, None]
            else: # Already (batch,1)
                self.y_task_target = self.y_task_target_original
        else: # For multi-bit output, ensure it's (batch, output_n)
            self.y_task_target = self.y_task_target_original[:, :OUTPUT_N]


    @staticmethod # Make it a static method
    @partial(nnx.jit, static_argnames=("static_topology_type", "static_gnn_steps", "is_training_static", "single_lut_shape_val"))
    def _compiled_gnn_update_and_eval(
        # No 'self'
        gnn_model_arg: CircuitGNN,                 # Explicitly pass GNN model
        single_lut_shape_val: Tuple,             # Explicitly pass single_lut_shape, now static
        initial_lut_logits_list_jax: List[jp.ndarray],
        x_task_data_jax: jp.ndarray,
        y_task_target_jax: jp.ndarray,
        static_topology_type: str,
        static_gnn_steps: int,
        is_training_static: bool
    ):
        # This function is JIT-compiled and contains the core GNN update and circuit evaluation.

        def loss_fn_for_grad(model_state: CircuitGNN, current_single_lut_shape: Tuple):
            # model_state is what nnx.value_and_grad expects as the first arg for differentiation
            # 1. Build graph
            temp_eval_output = evaluate_fixed_circuit(static_topology_type, initial_lut_logits_list_jax, x_task_data_jax, hard=False)
            initial_circuit_loss = binary_cross_entropy(temp_eval_output, y_task_target_jax)

            gnn_graph = build_fixed_topology_graph(
                static_topology_type,
                initial_lut_logits_list_jax,
                GNN_HIDDEN_DIM, # GNN_HIDDEN_DIM is a global constant
                current_circuit_loss=initial_circuit_loss
            )

            # 2. Run GNN
            # model_state here is gnn_model_arg
            updated_gnn_graph = run_gnn_scan(model_state, gnn_graph, static_gnn_steps)

            # 3. Extract LUTs
            final_lut_logits_list = extract_final_lut_logits(updated_gnn_graph, current_single_lut_shape)

            # 4. Evaluate Circuit (soft for loss, hard for accuracy)
            circuit_output_soft = evaluate_fixed_circuit(static_topology_type, final_lut_logits_list, x_task_data_jax, hard=False)
            circuit_output_hard = evaluate_fixed_circuit(static_topology_type, final_lut_logits_list, x_task_data_jax, hard=True)

            # 5. Calculate task loss and accuracy
            task_loss = binary_cross_entropy(circuit_output_soft, y_task_target_jax)
            accuracy = compute_accuracy(circuit_output_hard, y_task_target_jax)

            return task_loss, (accuracy, circuit_output_hard, final_lut_logits_list)

        # If training, compute gradients and apply updates
        if is_training_static:
            # Differentiate wrt the first argument (model_state, which is gnn_model_arg)
            (loss, aux_data), grads = nnx.value_and_grad(loss_fn_for_grad, argnums=0, has_aux=True)(gnn_model_arg, single_lut_shape_val)
        else: # Evaluation only
            loss, aux_data = loss_fn_for_grad(gnn_model_arg, single_lut_shape_val)
            grads = None

        return loss, aux_data, grads


    def run_one_training_or_evaluation_step(self):
        # Prepare JAX arrays from potentially numpy inputs
        initial_luts_jax = [jp.array(l) for l in self.initial_lut_logits_list]
        x_task_jax = jp.array(self.x_task_data)
        y_task_jax = jp.array(self.y_task_target)

        # Call the static JIT-compiled function
        loss_val, aux_data, grads = FixedCircuitGNNDemo._compiled_gnn_update_and_eval(
            self.gnn_model,                         # Pass GNN model instance
            self.single_lut_shape,                  # Pass single_lut_shape
            initial_lut_logits_list_jax=initial_luts_jax,
            x_task_data_jax=x_task_jax,
            y_task_target_jax=y_task_jax,
            static_topology_type=self.current_topology,
            static_gnn_steps=GNN_MESSAGE_PASSING_STEPS,
            is_training_static=self.is_training
        )

        if self.is_training and grads is not None:
            self.optimizer.update(grads) # Apply updates using the grads from the compiled function
            self.train_step_count += 1
        # If not training, or if grads were None, no update is applied.

        acc_val, circ_out_hard_jax, fin_luts_jax = aux_data

        self.current_loss = float(loss_val)
        self.current_accuracy = float(acc_val)
        if self.is_training: # Only log if it was a training step where updates were applied
          self.losses.append(self.current_loss)
          self.accuracies.append(self.current_accuracy)

        self.display_circuit_output_hard = np.array(circ_out_hard_jax)
        self.display_lut_truth_tables_probs = [
            np.array(jax.nn.sigmoid(luts.flatten())) for luts in fin_luts_jax
        ]


    def gui(self):
        imgui.set_next_item_width(150)
        _, self.is_training = imgui.checkbox("Is Training", self.is_training)
        imgui.same_line()
        if imgui.button("Step") or (self.is_training and self.train_step_count % EPOCHS_PER_UI_UPDATE == 0):
            self.run_one_training_or_evaluation_step()

        imgui.text(f"Train Step: {self.train_step_count}")
        imgui.text(f"Current Loss: {self.current_loss:.4f}")
        imgui.text(f"Current Accuracy: {self.current_accuracy:.4f}")

        # --- Topology and Task Selection ---
        imgui.set_next_item_width(150)
        changed_topo, selected_topo_idx = imgui.combo("Topology", self.available_topologies.index(self.current_topology), self.available_topologies)
        if changed_topo:
            self.current_topology = self.available_topologies[selected_topo_idx]
            # Optionally reset GNN or logs when topology changes significantly
            # self.losses, self.accuracies, self.train_step_count = [], [], 0

        imgui.set_next_item_width(150)
        try:
            current_task_idx = self.available_tasks.index(self.target_task_name)
        except ValueError: # Should not happen if available_tasks is correct
            current_task_idx = 0
        changed_task, selected_task_idx = imgui.combo("Target Task", current_task_idx, self.available_tasks)
        if changed_task:
            self.target_task_name = self.available_tasks[selected_task_idx]
            self._load_task_data()
            # self.losses, self.accuracies, self.train_step_count = [], [], 0

        # --- Plotting ---
        if self.losses and implot.begin_plot("Training Metrics", (-1, 150)):
            implot.setup_axes("GNN Update Step", "Value", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            implot.plot_line("Loss", np.array(self.losses, dtype=np.float32))
            implot.plot_line("Accuracy", np.array(self.accuracies, dtype=np.float32))
            implot.end_plot()

        imgui.separator_text(f"Circuit: {self.current_topology.upper()} for {self.target_task_name.upper()}")
        # --- Visualization of Circuit and LUTs ---
        # (Text-based for now, graphical could be added using ImGui drawing)
        for i, tt_probs_np in enumerate(self.display_lut_truth_tables_probs):
            imgui.text(f"LUT {LUT_NODE_IDS[i]} (Probs for inputs 00,01,10,11):")
            tt_str = "  " + ", ".join([f"{p:.2f} (H:{np.round(p)})" for p in tt_probs_np])
            imgui.text_unformatted(tt_str)

        imgui.separator_text(f"Circuit Output (Hard) vs. Target (for all {CASE_N} cases):")
        if hasattr(self, 'x_task_data') and hasattr(self, 'y_task_target'):
            cols = 4
            table_flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.borders_outer | imgui.TableFlags_.borders_inner_v | imgui.TableFlags_.row_bg
            if imgui.begin_table("output_table", cols, flags=table_flags):
                imgui.table_setup_column("Input")
                imgui.table_setup_column("Circuit Out")
                imgui.table_setup_column("Target Out")
                imgui.table_setup_column("Match?")
                imgui.table_headers_row()

                for i in range(CASE_N):
                    imgui.table_next_row()
                    input_case_str = "".join(map(str, np.array(self.x_task_data[i]).astype(int)))
                    out_val = int(self.display_circuit_output_hard[i,0]) if self.display_circuit_output_hard.ndim > 1 else int(self.display_circuit_output_hard[i])
                    target_val = int(self.y_task_target[i,0]) if self.y_task_target.ndim > 1 else int(self.y_task_target[i])

                    imgui.table_set_column_index(0)
                    imgui.text(input_case_str)
                    imgui.table_set_column_index(1)
                    imgui.text(str(out_val))
                    imgui.table_set_column_index(2)
                    imgui.text(str(target_val))
                    imgui.table_set_column_index(3)
                    if out_val == target_val:
                        imgui.text_colored((0.2,1.0,0.2,1.0), "MATCH")
                    else:
                        imgui.text_colored((1.0,0.2,0.2,1.0), "MISMATCH")
                imgui.end_table()

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure JAX is using the desired backend, e.g., CPU for quick tests if GPU is not primary
    # jax.config.update("jax_platform_name", "cpu") # Uncomment for CPU

    demo_instance = FixedCircuitGNNDemo()
    immapp.run(
        gui_function=demo_instance.gui,
        window_title="Fixed Circuits GNN Demo",
        window_size=(1000, 700),
        fps_idle=10, # Lower FPS when idle
        with_implot=True,
    )
import jax.numpy as jp
import numpy as np # For initial logits if not using jax.numpy directly for that

def get_n4_chain_circuit_spec(arity=2):
    input_n = 4
    # Node mapping: Inputs: 0,1,2,3. Gates: G0=4, G1=5, G2=6
    initial_logits_per_gate = np.zeros((1, 1, 2**arity), dtype=np.float32) # e.g., NOPs
    logits_chain = [
        jp.array(initial_logits_per_gate), # For G0
        jp.array(initial_logits_per_gate), # For G1
        jp.array(initial_logits_per_gate)  # For G2
    ]
    wires_chain = [
        jp.array([[0],[1]]),
        jp.array([[2],[4]]),
        jp.array([[3],[5]])
    ]
    # layer_sizes_for_build_graph might be implicitly defined by logits structure
    # or gen_circuit could return it too.
    # For build_graph: num_input_nodes = input_n
    return wires_chain, logits_chain, input_n

def get_n4_balanced_circuit_spec(arity=2):
    input_n = 4
    # Node mapping: Inputs: 0,1,2,3. Gates: G0=4, G1=5 (parallel), G2=6
    initial_logits_g0g1 = np.zeros((2, 1, 2**arity), dtype=np.float32)
    initial_logits_g2 = np.zeros((1, 1, 2**arity), dtype=np.float32)
    logits_balanced = [
        jp.array(initial_logits_g0g1),
        jp.array(initial_logits_g2)
    ]
    wires_balanced = [
        jp.array([[0, 2], [1, 3]]),
        jp.array([[4], [5]])
    ]
    return wires_balanced, logits_balanced, input_n
#!/usr/bin/env python3

import jax
import jax.numpy as jp
import jraph
from boolean_nca_cc.models.self_attention import run_self_attention_scan, CircuitSelfAttention
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
import flax.nnx as nnx

# Create a simple test
rng = jax.random.PRNGKey(42)
layer_sizes = generate_layer_sizes(input_n=4, output_n=2, arity=2, layer_n=2)
wires, logits = gen_circuit(rng, layer_sizes, arity=2)

graph = build_graph(
    logits=logits,
    wires=wires,
    input_n=4,
    arity=2,
    circuit_hidden_dim=16
)

model = CircuitSelfAttention(
    n_node=graph.nodes['logits'].shape[0],
    circuit_hidden_dim=16,
    arity=2,
    attention_dim=64,
    num_heads=4,
    num_layers=1,
    zero_init=False,
    rngs=nnx.Rngs(42)
)

# Test the scan function
final_graph, all_graphs = run_self_attention_scan(
    model=model,
    graph=graph,
    num_steps=3,
    knockout_pattern=None
)

print(f"Type of all_graphs: {type(all_graphs)}")
print(f"Length of all_graphs: {len(all_graphs)}")
print(f"Type of all_graphs[0]: {type(all_graphs[0])}")
print(f"Type of all_graphs[1]: {type(all_graphs[1])}")

# Try to access nodes
try:
    print(f"all_graphs[0].nodes keys: {list(all_graphs[0].nodes.keys())}")
    print(f"all_graphs[1].nodes keys: {list(all_graphs[1].nodes.keys())}")
except Exception as e:
    print(f"Error accessing nodes: {e}")
    print(f"all_graphs[1] content: {all_graphs[1]}") 
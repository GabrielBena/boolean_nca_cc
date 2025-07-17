"""
Level 2.1: Graph Construction Validation Tests

This module tests the graph construction functionality in utils/graph_builder.py.
Focuses on validating the conversion from boolean circuits to Jraph GraphsTuple representations.
"""

import pytest
import jax
import jax.numpy as jp
import jraph
from typing import List, Tuple

from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.circuits.model import gen_circuit, make_nops


class TestBasicGraphStructureValidation:
    """Test basic graph structure and validity."""

    def test_build_graph_creates_valid_graphstuple(self):
        """Test that build_graph returns a valid jraph.GraphsTuple."""
        # Create simple circuit: 2 inputs -> 2 gates -> 1 output
        key = jax.random.PRNGKey(42)
        layer_sizes = [(2, 1), (2, 1), (1, 1)]  # (nodes, group_size) per layer
        wires, logits = gen_circuit(key, layer_sizes, arity=2)
        
        # Build graph
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=2,
            arity=2,
            circuit_hidden_dim=16
        )
        
        # Verify it's a valid GraphsTuple
        assert isinstance(graph, jraph.GraphsTuple)
        assert graph.nodes is not None
        assert graph.senders is not None
        assert graph.receivers is not None
        assert graph.n_node is not None
        assert graph.n_edge is not None
        assert graph.globals is not None

    def test_node_count_matches_circuit_size(self):
        """Test that total node count matches expected circuit size."""
        # Circuit: 3 inputs -> 4 gates -> 2 gates -> 1 output
        key = jax.random.PRNGKey(42)
        layer_sizes = [(3, 1), (4, 1), (2, 1), (1, 1)]
        wires, logits = gen_circuit(key, layer_sizes, arity=2)
        
        expected_nodes = 3 + 4 + 2 + 1  # input + gate layers
        
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=3,
            arity=2,
            circuit_hidden_dim=16
        )
        
        assert graph.n_node[0] == expected_nodes
        assert len(graph.nodes["layer"]) == expected_nodes

    def test_edge_count_matches_connectivity_requirements(self):
        """Test that edge count matches expected connectivity."""
        # Simple circuit: 2 inputs -> 2 gates (arity=2)
        key = jax.random.PRNGKey(42)
        layer_sizes = [(2, 1), (2, 1)]
        wires, logits = gen_circuit(key, layer_sizes, arity=2)
        
        # Expected edges: 2 gates * 2 inputs per gate = 4 forward edges
        # With bidirectional: 4 forward + 4 backward = 8 total
        expected_forward_edges = 2 * 2  # gates * arity
        expected_total_edges = expected_forward_edges * 2  # bidirectional
        
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=2,
            arity=2,
            circuit_hidden_dim=16,
            bidirectional_edges=True
        )
        
        assert graph.n_edge[0] == expected_total_edges
        assert len(graph.senders) == expected_total_edges
        assert len(graph.receivers) == expected_total_edges

    def test_graph_dimensions_and_shapes(self):
        """Test that all graph components have correct shapes."""
        key = jax.random.PRNGKey(42)
        layer_sizes = [(2, 1), (3, 1), (1, 1)]
        wires, logits = gen_circuit(key, layer_sizes, arity=2)
        
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=2,
            arity=2,
            circuit_hidden_dim=16
        )
        
        total_nodes = 2 + 3 + 1  # input + gates
        
        # Check node feature shapes
        assert graph.nodes["layer"].shape == (total_nodes,)
        assert graph.nodes["group"].shape == (total_nodes,)
        assert graph.nodes["gate_id"].shape == (total_nodes,)
        assert graph.nodes["logits"].shape == (total_nodes, 4)  # 2^arity = 4
        assert graph.nodes["hidden"].shape == (total_nodes, 16)  # circuit_hidden_dim
        assert graph.nodes["layer_pe"].shape == (total_nodes, 16)
        assert graph.nodes["intra_layer_pe"].shape == (total_nodes, 16)
        assert graph.nodes["loss"].shape == (total_nodes,)
        
        # Check edge shapes
        expected_edges = (3 * 2 + 1 * 2) * 2  # bidirectional
        assert graph.senders.shape == (expected_edges,)
        assert graph.receivers.shape == (expected_edges,)
        
        # Check global shape
        assert graph.globals.shape == (2,)  # [loss_value, update_steps]


class TestNodeFeatureValidation:
    """Test that node features are correctly assigned."""

    def test_input_layer_nodes_have_correct_features(self):
        """Test input layer nodes (layer 0) have correct features."""
        key = jax.random.PRNGKey(42)
        layer_sizes = [(3, 1), (2, 1)]
        wires, logits = gen_circuit(key, layer_sizes, arity=2)
        
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=3,
            arity=2,
            circuit_hidden_dim=16
        )
        
        # Input nodes should be the first 3 nodes
        input_mask = graph.nodes["layer"] == 0
        input_indices = jp.where(input_mask)[0]
        
        assert len(input_indices) == 3
        
        # Check input node features
        for idx in input_indices:
            assert graph.nodes["layer"][idx] == 0
            assert graph.nodes["group"][idx] == 0
            assert graph.nodes["gate_id"][idx] == idx
            assert jp.allclose(graph.nodes["logits"][idx], 0.0)  # Inputs have no logits
            assert jp.allclose(graph.nodes["hidden"][idx], 0.0)  # Initialized to zeros
            assert jp.allclose(graph.nodes["loss"][idx], 0.0)  # Initialized to zeros

    def test_gate_layer_nodes_have_correct_features(self):
        """Test gate layer nodes (layer > 0) have correct features."""
        key = jax.random.PRNGKey(42)
        layer_sizes = [(2, 1), (3, 1), (1, 1)]
        wires, logits = gen_circuit(key, layer_sizes, arity=2)
        
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=2,
            arity=2,
            circuit_hidden_dim=16
        )
        
        # Gate nodes should have layer > 0
        gate_mask = graph.nodes["layer"] > 0
        gate_indices = jp.where(gate_mask)[0]
        
        assert len(gate_indices) == 4  # 3 + 1 gates
        
        # Check gate node features
        for idx in gate_indices:
            layer = graph.nodes["layer"][idx]
            assert layer > 0
            assert graph.nodes["gate_id"][idx] == idx
            # Logits should not be all zeros for gates
            assert not jp.allclose(graph.nodes["logits"][idx], 0.0)
            assert jp.allclose(graph.nodes["hidden"][idx], 0.0)  # Initialized to zeros
            assert jp.allclose(graph.nodes["loss"][idx], 0.0)  # Initialized to zeros

    def test_positional_encodings_are_properly_generated(self):
        """Test that positional encodings are correctly generated."""
        key = jax.random.PRNGKey(42)
        layer_sizes = [(2, 1), (2, 1)]
        wires, logits = gen_circuit(key, layer_sizes, arity=2)
        
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=2,
            arity=2,
            circuit_hidden_dim=16
        )
        
        # Check that positional encodings have correct shape
        assert graph.nodes["layer_pe"].shape == (4, 16)  # total_nodes, pe_dim
        assert graph.nodes["intra_layer_pe"].shape == (4, 16)
        
        # Check that positional encodings are not all zeros
        assert not jp.allclose(graph.nodes["layer_pe"], 0.0)
        assert not jp.allclose(graph.nodes["intra_layer_pe"], 0.0)
        
        # Check that different layers have different layer_pe
        input_layer_pe = graph.nodes["layer_pe"][:2]  # First 2 nodes (input layer)
        gate_layer_pe = graph.nodes["layer_pe"][2:]   # Last 2 nodes (gate layer)
        assert not jp.allclose(input_layer_pe, gate_layer_pe)

    def test_hidden_states_initialized_to_zeros(self):
        """Test that hidden states are properly initialized to zeros."""
        key = jax.random.PRNGKey(42)
        layer_sizes = [(2, 1), (3, 1)]
        wires, logits = gen_circuit(key, layer_sizes, arity=2)
        
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=2,
            arity=2,
            circuit_hidden_dim=16
        )
        
        # All hidden states should be initialized to zeros
        assert jp.allclose(graph.nodes["hidden"], 0.0)
        assert graph.nodes["hidden"].shape == (5, 16)  # total_nodes, hidden_dim


class TestConnectivityPatternValidation:
    """Test that edge connectivity patterns are correct."""

    def test_forward_edges_connect_previous_to_current_layers(self):
        """Test that forward edges properly connect layers."""
        key = jax.random.PRNGKey(42)
        layer_sizes = [(2, 1), (2, 1)]
        wires, logits = gen_circuit(key, layer_sizes, arity=2)
        
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=2,
            arity=2,
            circuit_hidden_dim=16,
            bidirectional_edges=False  # Only forward edges for clarity
        )
        
        # Forward edges: senders should be from previous layer, receivers from current
        for i in range(len(graph.senders)):
            sender = graph.senders[i]
            receiver = graph.receivers[i]
            
            sender_layer = graph.nodes["layer"][sender]
            receiver_layer = graph.nodes["layer"][receiver]
            
            # Sender should be from previous layer
            assert sender_layer == receiver_layer - 1

    def test_bidirectional_edges_create_proper_sender_receiver_pairs(self):
        """Test that bidirectional edges create correct forward/backward pairs."""
        key = jax.random.PRNGKey(42)
        layer_sizes = [(2, 1), (2, 1)]
        wires, logits = gen_circuit(key, layer_sizes, arity=2)
        
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=2,
            arity=2,
            circuit_hidden_dim=16,
            bidirectional_edges=True
        )
        
        # Should have 4 forward + 4 backward = 8 edges
        assert len(graph.senders) == 8
        assert len(graph.receivers) == 8
        
        # Check that backward edges are reverse of forward edges
        forward_senders = graph.senders[:4]
        forward_receivers = graph.receivers[:4]
        backward_senders = graph.senders[4:]
        backward_receivers = graph.receivers[4:]
        
        # Backward senders should match forward receivers
        assert jp.array_equal(backward_senders, forward_receivers)
        # Backward receivers should match forward senders
        assert jp.array_equal(backward_receivers, forward_senders)

    def test_wire_indices_map_correctly_to_global_node_indices(self):
        """Test that wire connection patterns map correctly to global indices."""
        key = jax.random.PRNGKey(42)
        layer_sizes = [(2, 1), (2, 1)]
        wires, logits = gen_circuit(key, layer_sizes, arity=2)
        
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=2,
            arity=2,
            circuit_hidden_dim=16,
            bidirectional_edges=False
        )
        
        # wires[0] connects input layer (nodes 0,1) to first gate layer (nodes 2,3)
        # Each gate in layer 1 has 2 inputs (arity=2)
        layer_wires = wires[0]  # Shape: (arity, num_gates) = (2, 2)
        
        # Check that wire indices are within valid range for input layer
        assert jp.all(layer_wires < 2)  # Input layer has 2 nodes (indices 0,1)
        
        # Check that edges connect to correct input nodes
        forward_edges = list(zip(graph.senders, graph.receivers))
        for sender, receiver in forward_edges:
            if receiver >= 2:  # Gate node
                # Sender should be a valid input node
                assert sender < 2
                # Receiver should be a gate node
                assert receiver >= 2

    def test_edge_connectivity_matches_circuit_wiring(self):
        """Test that edge connectivity matches the original circuit wiring."""
        key = jax.random.PRNGKey(42)
        layer_sizes = [(2, 1), (2, 1)]
        wires, logits = gen_circuit(key, layer_sizes, arity=2)
        
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=2,
            arity=2,
            circuit_hidden_dim=16,
            bidirectional_edges=False
        )
        
        # Extract forward edges only
        forward_edges = list(zip(graph.senders, graph.receivers))
        
        # Each gate should have exactly 'arity' input connections
        gate_connections = {}
        for sender, receiver in forward_edges:
            receiver_int = int(receiver)  # Convert JAX array to int for dict key
            if receiver_int not in gate_connections:
                gate_connections[receiver_int] = []
            gate_connections[receiver_int].append(int(sender))  # Convert to int
        
        # Check that each gate has correct number of inputs
        for gate_node in [2, 3]:  # Gate nodes are at indices 2,3
            assert len(gate_connections[gate_node]) == 2  # arity=2
            # All connections should be from input layer (nodes 0,1)
            for input_node in gate_connections[gate_node]:
                assert input_node < 2


if __name__ == "__main__":
    pytest.main([__file__]) 
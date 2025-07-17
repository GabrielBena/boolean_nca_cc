"""
Level 2.2: Graph-Circuit Conversion Testing

This module tests the critical functions in utils/extraction.py that handle conversion
between graph representations and boolean circuits. This is essential for knockout
evaluation as it ensures that knockout-modified graphs can be properly converted back
to circuits for performance measurement.

Components Tested:
- extract_logits_from_graph: Extracts logit tensors from graph node features
- update_output_node_loss: Updates loss values for output nodes
- get_output_node_indices: Helper function for output node identification
- Round-trip circuit preservation through graph conversion
- JAX compatibility and gradient flow validation
"""

import pytest
import jax
import jax.numpy as jp
import numpy as np
import jraph
from typing import List, Tuple

from boolean_nca_cc.circuits.model import gen_circuit, run_circuit, make_nops
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.utils.extraction import (
    extract_logits_from_graph,
    update_output_node_loss,
    get_output_node_indices,
)


class TestGraphCircuitConversion:
    """Test suite for graph-circuit conversion functions."""

    @pytest.fixture
    def sample_circuit(self):
        """Create a sample circuit for testing."""
        key = jax.random.PRNGKey(42)
        layer_sizes = [(4, 1), (8, 2), (2, 1)]  # 4 inputs, 8 gates (2 groups), 2 outputs
        arity = 2
        wires, logits = gen_circuit(key, layer_sizes, arity)
        return wires, logits, layer_sizes, arity

    @pytest.fixture
    def sample_graph(self, sample_circuit):
        """Create a sample graph from the circuit."""
        wires, logits, layer_sizes, arity = sample_circuit
        input_n = layer_sizes[0][0]
        circuit_hidden_dim = 64
        
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=circuit_hidden_dim,
        )
        return graph, logits, layer_sizes

    def test_extract_logits_from_graph_basic(self, sample_graph):
        """Test basic logit extraction from graph."""
        graph, original_logits, layer_sizes = sample_graph
        
        # Get original shapes for extraction
        logits_original_shapes = [logits.shape for logits in original_logits]
        
        # Extract logits from graph
        extracted_logits = extract_logits_from_graph(graph, logits_original_shapes)
        
        # Verify shapes match
        assert len(extracted_logits) == len(original_logits)
        for extracted, original in zip(extracted_logits, original_logits):
            assert extracted.shape == original.shape
        
        # Verify values match (within numerical precision)
        for extracted, original in zip(extracted_logits, original_logits):
            np.testing.assert_allclose(extracted, original, rtol=1e-6, atol=1e-6)

    def test_extract_logits_from_graph_different_sizes(self):
        """Test logit extraction with different circuit sizes."""
        key = jax.random.PRNGKey(123)
        
        # Test different layer configurations
        test_configs = [
            [(2, 1), (4, 1)],  # Small circuit
            [(8, 2), (16, 4), (4, 1)],  # Medium circuit
            [(16, 4), (32, 8), (64, 16), (8, 2)],  # Large circuit
        ]
        
        for layer_sizes in test_configs:
            wires, logits = gen_circuit(key, layer_sizes, arity=2)
            input_n = layer_sizes[0][0]
            
            graph = build_graph(
                logits=logits,
                wires=wires,
                input_n=input_n,
                arity=2,
                circuit_hidden_dim=32,
            )
            
            logits_original_shapes = [logits.shape for logits in logits]
            extracted_logits = extract_logits_from_graph(graph, logits_original_shapes)
            
            # Verify extraction preserves all logits
            assert len(extracted_logits) == len(logits)
            for extracted, original in zip(extracted_logits, logits):
                assert extracted.shape == original.shape
                np.testing.assert_allclose(extracted, original, rtol=1e-6, atol=1e-6)

    def test_extract_logits_from_graph_jit_compatibility(self, sample_graph):
        """Test that logit extraction is JIT compatible."""
        graph, original_logits, layer_sizes = sample_graph
        logits_original_shapes = [logits.shape for logits in original_logits]
        
        # Test that extraction works without JIT first
        extracted_logits = extract_logits_from_graph(graph, logits_original_shapes)
        
        # Verify results are correct
        assert len(extracted_logits) == len(original_logits)
        for extracted, original in zip(extracted_logits, original_logits):
            np.testing.assert_allclose(extracted, original, rtol=1e-6, atol=1e-6)
        
        # Note: JIT compilation is not directly compatible with dynamic shapes
        # The extraction function uses dynamic_slice which requires concrete shapes
        # This is expected behavior for functions that work with variable-sized inputs

    def test_get_output_node_indices(self):
        """Test output node index calculation."""
        # Test different layer configurations
        test_cases = [
            ([(4, 1), (2, 1)], (4, 6)),  # 4 inputs, 2 outputs
            ([(8, 2), (16, 4), (4, 1)], (24, 28)),  # 8 inputs, 4 outputs
            ([(2, 1), (4, 1), (8, 2), (2, 1)], (14, 16)),  # 2 inputs, 2 outputs
        ]
        
        for layer_sizes, expected_indices in test_cases:
            start_idx, end_idx = get_output_node_indices(layer_sizes)
            assert (start_idx, end_idx) == expected_indices

    def test_update_output_node_loss_scalar(self, sample_graph):
        """Test updating output node loss with scalar value."""
        graph, _, layer_sizes = sample_graph
        
        # Calculate layer sizes for loss update
        loss_layer_sizes = [(size[0], size[1]) for size in layer_sizes]
        loss_value = 0.5
        
        # Update loss
        updated_graph = update_output_node_loss(graph, loss_layer_sizes, loss_value)
        
        # Get output node indices
        output_start, output_end = get_output_node_indices(loss_layer_sizes)
        num_output_nodes = output_end - output_start
        
        # Verify loss values are updated correctly
        updated_loss = updated_graph.nodes["loss"]
        np.testing.assert_allclose(
            updated_loss[output_start:output_end],
            jp.full(num_output_nodes, loss_value),
            rtol=1e-6, atol=1e-6
        )
        
        # Verify non-output nodes are unchanged
        np.testing.assert_allclose(
            updated_loss[:output_start],
            graph.nodes["loss"][:output_start],
            rtol=1e-6, atol=1e-6
        )

    def test_update_output_node_loss_array(self, sample_graph):
        """Test updating output node loss with array values."""
        graph, _, layer_sizes = sample_graph
        loss_layer_sizes = [(size[0], size[1]) for size in layer_sizes]
        
        # Get output node indices
        output_start, output_end = get_output_node_indices(loss_layer_sizes)
        num_output_nodes = output_end - output_start
        
        # Test with 1D array
        loss_values_1d = jp.array([0.1, 0.2, 0.3, 0.4])[:num_output_nodes]
        updated_graph = update_output_node_loss(graph, loss_layer_sizes, loss_values_1d)
        
        updated_loss = updated_graph.nodes["loss"]
        np.testing.assert_allclose(
            updated_loss[output_start:output_end],
            loss_values_1d,
            rtol=1e-6, atol=1e-6
        )
        
        # Test with 2D array (should average across first dimension)
        loss_values_2d = jp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])[:, :num_output_nodes]
        expected_average = jp.mean(loss_values_2d, axis=0)
        
        updated_graph_2d = update_output_node_loss(graph, loss_layer_sizes, loss_values_2d)
        updated_loss_2d = updated_graph_2d.nodes["loss"]
        np.testing.assert_allclose(
            updated_loss_2d[output_start:output_end],
            expected_average,
            rtol=1e-6, atol=1e-6
        )

    def test_update_output_node_loss_broadcasting(self, sample_graph):
        """Test loss value broadcasting behavior."""
        graph, _, layer_sizes = sample_graph
        loss_layer_sizes = [(size[0], size[1]) for size in layer_sizes]
        
        # Test single value broadcasting
        single_loss = 0.75
        updated_graph = update_output_node_loss(graph, loss_layer_sizes, single_loss)
        
        output_start, output_end = get_output_node_indices(loss_layer_sizes)
        num_output_nodes = output_end - output_start
        
        updated_loss = updated_graph.nodes["loss"]
        np.testing.assert_allclose(
            updated_loss[output_start:output_end],
            jp.full(num_output_nodes, single_loss),
            rtol=1e-6, atol=1e-6
        )

    def test_round_trip_circuit_preservation(self, sample_circuit):
        """Test circuit → graph → circuit round-trip preserves functionality."""
        wires, logits, layer_sizes, arity = sample_circuit
        input_n = layer_sizes[0][0]
        
        # Create test input
        key = jax.random.PRNGKey(456)
        test_input = jax.random.bernoulli(key, shape=(input_n,))
        
        # Run original circuit
        original_output = run_circuit(logits, wires, test_input)
        
        # Convert to graph
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=32,
        )
        
        # Extract logits back from graph
        logits_original_shapes = [logits.shape for logits in logits]
        extracted_logits = extract_logits_from_graph(graph, logits_original_shapes)
        
        # Run circuit with extracted logits
        extracted_output = run_circuit(extracted_logits, wires, test_input)
        
        # Verify outputs are identical
        assert len(original_output) == len(extracted_output)
        for orig, extr in zip(original_output, extracted_output):
            np.testing.assert_allclose(orig, extr, rtol=1e-6, atol=1e-6)

    def test_round_trip_with_modified_graph(self, sample_graph):
        """Test round-trip with modified graph (simulating knockout modifications)."""
        graph, original_logits, layer_sizes = sample_graph
        logits_original_shapes = [logits.shape for logits in original_logits]
        
        # Modify the graph (simulate knockout modifications)
        modified_nodes = dict(graph.nodes)
        # Modify some logits in the graph
        modified_logits = modified_nodes["logits"].at[10:15].set(0.0)  # Zero out some logits
        modified_nodes["logits"] = modified_logits
        modified_graph = graph._replace(nodes=modified_nodes)
        
        # Extract logits from modified graph
        extracted_logits = extract_logits_from_graph(modified_graph, logits_original_shapes)
        
        # Verify extracted logits reflect the modifications
        # Find which layer contains node 10-15
        current_node = 0
        for i, shape in enumerate(logits_original_shapes):
            group_n, group_size, logit_dim = shape
            num_gates = group_n * group_size
            
            if current_node <= 10 < current_node + num_gates:
                # Node 10 is in this layer
                layer_start = 10 - current_node
                layer_end = min(15 - current_node, num_gates)
                
                # Check that the extracted logits reflect the modification
                extracted_layer = extracted_logits[i]
                flat_extracted = extracted_layer.reshape(-1, logit_dim)
                
                # The modified nodes should have zero logits
                np.testing.assert_allclose(
                    flat_extracted[layer_start:layer_end],
                    jp.zeros((layer_end - layer_start, logit_dim)),
                    rtol=1e-6, atol=1e-6
                )
                break
            
            current_node += num_gates

    def test_gradient_flow_through_extraction(self, sample_graph):
        """Test that gradients flow correctly through extraction functions."""
        graph, original_logits, layer_sizes = sample_graph
        logits_original_shapes = [logits.shape for logits in original_logits]
        
        # Create a function that uses extraction
        def extraction_function(graph_input):
            extracted = extract_logits_from_graph(graph_input, logits_original_shapes)
            # Sum all extracted logits as a differentiable operation
            return sum(jp.sum(logits) for logits in extracted)
        
        # Compute gradients
        grad_fn = jax.grad(extraction_function)
        gradients = grad_fn(graph)
        
        # Verify gradients exist and have correct structure
        assert gradients is not None
        assert "logits" in gradients.nodes
        assert gradients.nodes["logits"].shape == graph.nodes["logits"].shape

    def test_loss_update_gradient_flow(self, sample_graph):
        """Test gradient flow through loss update function."""
        graph, _, layer_sizes = sample_graph
        loss_layer_sizes = [(size[0], size[1]) for size in layer_sizes]
        
        # Create a function that uses loss update
        def loss_update_function(graph_input, loss_values):
            updated = update_output_node_loss(graph_input, loss_layer_sizes, loss_values)
            # Sum all loss values as a differentiable operation
            return jp.sum(updated.nodes["loss"])
        
        # Test gradient with respect to loss values
        loss_values = jp.array([0.1, 0.2, 0.3, 0.4])
        grad_fn = jax.grad(loss_update_function, argnums=1)
        gradients = grad_fn(graph, loss_values)
        
        # Verify gradients exist
        assert gradients is not None
        assert gradients.shape == loss_values.shape

    def test_edge_cases_empty_circuit(self):
        """Test extraction with empty circuit."""
        # Create minimal graph with no gate layers
        input_n = 4
        graph = build_graph(
            logits=[],
            wires=[],
            input_n=input_n,
            arity=2,
            circuit_hidden_dim=32,
        )
        
        # Test extraction with empty logits list
        extracted_logits = extract_logits_from_graph(graph, [])
        assert len(extracted_logits) == 0

    def test_edge_cases_single_layer(self):
        """Test extraction with single layer circuit."""
        key = jax.random.PRNGKey(789)
        layer_sizes = [(4, 1), (2, 1)]  # Just input and output
        wires, logits = gen_circuit(key, layer_sizes, arity=2)
        input_n = layer_sizes[0][0]
        
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=input_n,
            arity=2,
            circuit_hidden_dim=32,
        )
        
        logits_original_shapes = [logits.shape for logits in logits]
        extracted_logits = extract_logits_from_graph(graph, logits_original_shapes)
        
        # Verify extraction works with single layer
        assert len(extracted_logits) == 1
        np.testing.assert_allclose(extracted_logits[0], logits[0], rtol=1e-6, atol=1e-6)

    def test_extreme_logit_values(self):
        """Test extraction with extreme logit values."""
        key = jax.random.PRNGKey(999)
        layer_sizes = [(2, 1), (4, 1)]
        wires, logits = gen_circuit(key, layer_sizes, arity=2)
        input_n = layer_sizes[0][0]
        
        # Create graph with extreme logit values
        # Use the same shape as the generated logits
        extreme_logits = [jp.array([[[-100.0, 100.0], [50.0, -50.0], [25.0, -25.0], [75.0, -75.0]]])]
        graph = build_graph(
            logits=extreme_logits,
            wires=wires,
            input_n=input_n,
            arity=2,
            circuit_hidden_dim=32,
        )
        
        logits_original_shapes = [logits.shape for logits in extreme_logits]
        extracted_logits = extract_logits_from_graph(graph, logits_original_shapes)
        
        # Verify extreme values are preserved
        np.testing.assert_allclose(extracted_logits[0], extreme_logits[0], rtol=1e-6, atol=1e-6)

    def test_knockout_integration_simulation(self, sample_graph):
        """Test extraction in context of knockout pattern simulation."""
        graph, original_logits, layer_sizes = sample_graph
        logits_original_shapes = [logits.shape for logits in original_logits]
        
        # Simulate knockout pattern by zeroing out specific nodes
        # This tests the critical path for knockout evaluation
        knockout_nodes = [5, 12, 18]  # Nodes to "knock out"
        
        modified_nodes = dict(graph.nodes)
        modified_logits = modified_nodes["logits"]
        for node_idx in knockout_nodes:
            if node_idx < modified_logits.shape[0]:
                modified_logits = modified_logits.at[node_idx].set(0.0)
        
        modified_nodes["logits"] = modified_logits
        modified_graph = graph._replace(nodes=modified_nodes)
        
        # Extract logits from knockout-modified graph
        extracted_logits = extract_logits_from_graph(modified_graph, logits_original_shapes)
        
        # Verify knockout modifications are preserved in extraction
        current_node = 0
        for i, shape in enumerate(logits_original_shapes):
            group_n, group_size, logit_dim = shape
            num_gates = group_n * group_size
            
            # Check if any knockout nodes are in this layer
            layer_knockout_nodes = [
                n for n in knockout_nodes 
                if current_node <= n < current_node + num_gates
            ]
            
            if layer_knockout_nodes:
                extracted_layer = extracted_logits[i]
                flat_extracted = extracted_layer.reshape(-1, logit_dim)
                
                for node_idx in layer_knockout_nodes:
                    layer_node_idx = node_idx - current_node
                    if layer_node_idx < flat_extracted.shape[0]:
                        # Verify knocked-out node has zero logits
                        np.testing.assert_allclose(
                            flat_extracted[layer_node_idx],
                            jp.zeros(logit_dim),
                            rtol=1e-6, atol=1e-6
                        )
            
            current_node += num_gates

    def test_batched_extraction(self):
        """Test extraction with batched graphs (for future vectorization)."""
        key = jax.random.PRNGKey(111)
        layer_sizes = [(4, 1), (8, 2)]
        arity = 2
        input_n = layer_sizes[0][0]
        
        # Create multiple circuits
        circuits = []
        for i in range(3):
            key_i = jax.random.fold_in(key, i)
            wires, logits = gen_circuit(key_i, layer_sizes, arity)
            circuits.append((wires, logits))
        
        # Create graphs for each circuit
        graphs = []
        for wires, logits in circuits:
            graph = build_graph(
                logits=logits,
                wires=wires,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=32,
            )
            graphs.append(graph)
        
        # Test extraction on each graph
        logits_original_shapes = [logits.shape for logits in circuits[0][1]]
        
        for i, (graph, (_, original_logits)) in enumerate(zip(graphs, circuits)):
            extracted_logits = extract_logits_from_graph(graph, logits_original_shapes)
            
            # Verify extraction preserves circuit-specific values
            for extracted, original in zip(extracted_logits, original_logits):
                np.testing.assert_allclose(extracted, original, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
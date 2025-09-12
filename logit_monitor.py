#!/usr/bin/env python3
"""
Logit channel monitoring utilities for training.

This module provides utilities to monitor logit channels during training,
working around JAX's JIT compilation limitations.
"""

import jax
import jax.numpy as jp
import logging
from functools import partial

log = logging.getLogger(__name__)


def logit_monitor_fn(graph, step, prefix=""):
    """
    Monitor function that can be called from JAX code to log logit information.
    
    This function is designed to work with JAX's jax.debug.print or similar mechanisms.
    """
    logits = graph.nodes["logits"]
    
    # Use jax.debug.print to print from within JIT-compiled code
    jax.debug.print(f"{prefix}Step {step} - Logits shape: {logits.shape}")
    jax.debug.print(f"{prefix}Logits range: [{jp.min(logits):.3f}, {jp.max(logits):.3f}]")
    jax.debug.print(f"{prefix}Logits mean: {jp.mean(logits):.3f}")
    
    # Check for damaged nodes (logits ≈ -10)
    damaged_mask = jp.abs(logits + 10.0) < 1e-6
    num_damaged = jp.sum(damaged_mask)
    jax.debug.print(f"{prefix}Damaged nodes (logits ≈ -10): {num_damaged}")
    
    return graph


def create_monitored_model_step(model, monitor_fn=None):
    """
    Create a monitored version of the model's __call__ method.
    
    Args:
        model: The CircuitSelfAttention model
        monitor_fn: Optional monitoring function to call after each step
    
    Returns:
        Monitored version of the model call
    """
    original_call = model.__call__
    
    def monitored_call(graph, **kwargs):
        # Call the original model
        result = original_call(graph, **kwargs)
        
        # Apply monitoring if provided
        if monitor_fn is not None:
            result = monitor_fn(result, step=0, prefix="[MONITOR] ")
        
        return result
    
    return monitored_call


def add_logit_logging_to_training():
    """
    Example of how to add logit logging to the training loop.
    
    This would be integrated into the train_loop.py file.
    """
    # Example usage in train_loop.py:
    """
    # In the pool_train_step function, add monitoring:
    
    def monitored_loss_fn_no_scan(model, graph, logits, wires, loss_key, knockout_pattern):
        # ... existing code ...
        
        for i in range(n_message_steps):
            graph = model(
                graph,
                knockout_pattern=knockout_pattern,
            )
            
            # Add logit monitoring here
            if i == 0:  # Only log first step to avoid spam
                graph = logit_monitor_fn(graph, i, f"[TRAIN] ")
            
            # ... rest of existing code ...
    """
    pass


# Example of how to use this in a training script
def example_usage():
    """Example of how to use the logit monitoring."""
    
    # Method 1: Using jax.debug.print (works in JIT)
    @jax.jit
    def monitored_step(graph, model, knockout_pattern):
        # Apply model
        result = model(graph, knockout_pattern=knockout_pattern)
        
        # Monitor logits
        logits = result.nodes["logits"]
        jax.debug.print("Logits shape: {}", logits.shape)
        jax.debug.print("Logits range: [{}, {}]", jp.min(logits), jp.max(logits))
        
        # Check for damaged nodes
        damaged = jp.sum(jp.abs(logits + 10.0) < 1e-6)
        jax.debug.print("Damaged nodes: {}", damaged)
        
        return result
    
    # Method 2: Using a callback (requires non-JIT version)
    def non_jit_monitored_step(graph, model, knockout_pattern):
        result = model(graph, knockout_pattern=knockout_pattern)
        logit_monitor_fn(result, step=0, prefix="[CALLBACK] ")
        return result
    
    return monitored_step, non_jit_monitored_step


if __name__ == "__main__":
    print("Logit monitoring utilities loaded.")
    print("To use in training, integrate the monitoring functions into train_loop.py")
    print("Example usage:")
    example_usage()

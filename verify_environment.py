#!/usr/bin/env python3
"""Verify environment setup for Boolean NCA CC project."""

import sys
import jax
import jaxlib
import flax
import numpy as np

def verify_environment():
    print("=== Environment Verification ===")
    
    # Check versions
    print(f"Python: {sys.version}")
    print(f"JAX: {jax.__version__}")
    print(f"JAXlib: {jaxlib.__version__}")
    print(f"Flax: {flax.__version__}")
    
    # Check devices
    devices = jax.devices()
    print(f"Available devices: {devices}")
    print(f"Default backend: {jax.default_backend()}")
    
    # Test computation
    x = jax.numpy.array([1., 2., 3.])
    y = jax.numpy.sin(x)
    print(f"Test computation: {y}")
    
    # Test GPU memory
    if jax.default_backend() == 'gpu':
        try:
            large_array = jax.numpy.random.normal(jax.random.PRNGKey(0), (1000, 1000))
            result = jax.numpy.linalg.svd(large_array)
            print("GPU memory test: PASSED")
        except Exception as e:
            print(f"GPU memory test: FAILED - {e}")
    
    print("=== Verification Complete ===")

if __name__ == "__main__":
    verify_environment() 
# Environment Setup Guide for Boolean NCA CC

This guide provides step-by-step instructions to reproduce the exact environment used for the Boolean NCA CC project on a new machine.

## System Requirements

- **OS**: Linux (tested on Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with CUDA 12.5+ support
- **Memory**: 16GB+ RAM recommended
- **Storage**: 50GB+ free space

## Current Environment Specifications

- **Python**: 3.13.3
- **JAX**: 0.6.1 (with CUDA 12 support)
- **JAXlib**: 0.6.1
- **Flax**: 0.10.6
- **CUDA**: 12.9
- **cuDNN**: 9.8.0.87
- **NCCL**: 2.26.5.1
- **GPU**: 4x NVIDIA RTX A6000 (48GB each)

## Step 1: Install NVIDIA Drivers and CUDA

### 1.1 Install NVIDIA Drivers
```bash
# Check current GPU
nvidia-smi

# If drivers not installed, install them:
sudo apt update
sudo apt install nvidia-driver-555  # or latest version
sudo reboot
```

### 1.2 Install CUDA Toolkit 12.5+
```bash
# Download CUDA 12.5 from NVIDIA website
wget https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda_12.5.0_535.86.10_linux.run

# Install CUDA
sudo sh cuda_12.5.0_535.86.10_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.5/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Step 2: Install Miniconda

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# Initialize conda
conda init bash
source ~/.bashrc
```

## Step 3: Create and Setup Environment

### 3.1 Create Environment
```bash
# Create new environment
conda create -n metabool python=3.13.3

# Activate environment
conda activate metabool
```

### 3.2 Install Core Dependencies
```bash
# Install core packages via conda
conda install -c conda-forge numpy=2.2.5 scipy=1.15.2 matplotlib=3.10.1

# Install JAX with CUDA support
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install Flax
pip install flax==0.10.6

# Install other ML libraries
pip install optax==0.2.4 orbax-checkpoint==0.6.4 chex==0.1.88
```

### 3.3 Install Project Dependencies
```bash
# Install Hydra for configuration management
pip install hydra-core==1.3.2 omegaconf==2.3.0

# Install WandB for experiment tracking
pip install wandb==0.19.11

# Install Jupyter for development
pip install jupyter jupyterlab ipykernel

# Install other utilities
pip install tqdm rich pandas
```

## Step 4: Verify Installation

### 4.1 Test JAX Installation
```python
import jax
import jaxlib
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"JAXlib version: {jaxlib.__version__}")
print(f"Available devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Test GPU computation
x = jnp.array([1., 2., 3.])
y = jnp.sin(x)
print(f"Computation result: {y}")
```

### 4.2 Test Flax Installation
```python
import flax
from flax import nnx

print(f"Flax version: {flax.__version__}")

# Test basic model creation
rng = nnx.Rngs(params=jax.random.PRNGKey(0))
print("Flax NNX working correctly")
```

## Step 5: Alternative - Use Environment File

If the above step-by-step approach doesn't work, you can try using the exported environment file:

### 5.1 Using the environment.yml file
```bash
# Create environment from file
conda env create -f environment.yml

# Activate
conda activate metabool
```

**Note**: The environment.yml file contains exact package versions but may have platform-specific dependencies that need adjustment.

## Step 6: Install Project Package

```bash
# Navigate to project directory
cd boolean_nca_cc

# Install in development mode
pip install -e .
```

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   ```bash
   # Check CUDA version
   nvcc --version
   
   # If mismatch, reinstall JAX with correct CUDA version
   pip uninstall jax jaxlib
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

2. **Memory Issues**
   ```bash
   # Set JAX memory growth
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export XLA_PYTHON_CLIENT_MEM_FRACTION=.8
   ```

3. **GPU Not Detected**
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # Check JAX devices
   python -c "import jax; print(jax.devices())"
   ```

### JAX Fundamentals for Understanding

JAX is a high-performance numerical computing library that provides:
- **Automatic differentiation**: `jax.grad()` for gradients
- **Vectorization**: `jax.vmap()` for batch operations  
- **Just-in-time compilation**: `jax.jit()` for optimized execution
- **GPU/TPU acceleration**: Automatic device placement

The key JAX concepts in your project:
- **Transformations**: Functions that transform other functions (grad, jit, vmap)
- **Device arrays**: Immutable arrays that can be placed on different devices
- **Functional programming**: Pure functions without side effects
- **XLA compilation**: JAX uses XLA to compile and optimize computations

## Verification Script

Create a verification script to test your setup:

```python
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
```

Run this script to verify your installation:
```bash
python verify_environment.py
```

## Next Steps

1. Clone your project repository
2. Install the project package: `pip install -e .`
3. Run your training scripts
4. Monitor GPU usage with `nvidia-smi`

This setup should provide a reproducible environment matching your current metabool conda environment. 
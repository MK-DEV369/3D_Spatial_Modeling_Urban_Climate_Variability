# JAX GPU Installation Guide for Windows with CUDA 13.1 + RTX 3070

## Problem
JAX 0.8.2+ changed how CUDA installation works. The old `jax[cuda12_pip]` syntax no longer works on Windows.

## Solution

### Method 1: Install JAX with CUDA 12 support (Recommended)

JAX now requires separate CUDA plugin installation:

```bash
# 1. Uninstall current JAX
pip uninstall jax jaxlib -y

# 2. Install JAX with CUDA 12 support (compatible with CUDA 13.1)
pip install jax[cuda12]
```

**Important:** This installs CUDA libraries via pip (doesn't rely on system CUDA).

---

### Method 2: Install specific JAX version with working CUDA

Use an older JAX version that has stable Windows CUDA support:

```bash
# Uninstall current version
pip uninstall jax jaxlib -y

# Install JAX 0.4.23 (last version with reliable Windows CUDA support)
pip install jax[cuda12]==0.4.23 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

### Method 3: WSL2 (Best Performance - Recommended if having issues)

JAX GPU support on Windows can be problematic. WSL2 gives native Linux performance:

```bash
# In PowerShell (as Administrator)
wsl --install

# Then in WSL2 Ubuntu:
cd /mnt/e/5th\ SEM\ Data/MainEL/Final_Project/backend
python3 -m venv venv
source venv/bin/activate
pip install jax[cuda12]
```

---

## Quick Test Commands

After installation, test with:

```bash
# Test 1: Check backend
python -c "import jax; print('Backend:', jax.default_backend())"

# Test 2: Check devices
python -c "import jax; print('Devices:', jax.devices())"

# Test 3: Run computation
python test_gpu.py
```

**Expected Output:**
```
Backend: gpu
Devices: [cuda(id=0)]
```

---

## Official Resources

1. **JAX Installation Guide:**
   https://jax.readthedocs.io/en/latest/installation.html

2. **JAX CUDA Installation:**
   https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier

3. **JAX Windows Support:**
   https://github.com/google/jax/discussions/14510

4. **CUDA Compatibility:**
   https://jax.readthedocs.io/en/latest/installation.html#supported-platforms

---

## Troubleshooting

### Issue: "Backend: cpu" instead of "gpu"

**Cause:** JAX didn't detect CUDA libraries

**Solutions:**

1. Try Method 2 (older JAX version)
2. Check CUDA environment variables:
   ```powershell
   $env:PATH -split ';' | Select-String cuda
   ```

3. Install CUDA 12.x explicitly:
   - Download: https://developer.nvidia.com/cuda-12-0-0-download-archive
   - Or use conda: `conda install cuda -c nvidia/label/cuda-12.0.0`

### Issue: "jaxlib not found"

Your venv might not be properly activated. Check:
```bash
# Should show your venv path
python -c "import sys; print(sys.prefix)"
```

### Issue: Windows JAX GPU is slow/buggy

**Best solution:** Use WSL2 for better Linux compatibility

---

## Recommended Approach for Your Setup

Given your CUDA 13.1 and RTX 3070, I recommend:

### Option A: Try Latest JAX with CUDA 12 (Simplest)
```bash
pip uninstall jax jaxlib -y
pip install jax[cuda12]
python test_gpu.py
```

### Option B: Use Stable Version (Most Reliable)
```bash
pip uninstall jax jaxlib -y
pip install jax[cuda12]==0.4.23 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python test_gpu.py
```

### Option C: Switch to PyTorch or TensorFlow (Alternative)

If JAX continues to have issues on Windows:

```bash
# TensorFlow with GPU
pip install tensorflow[and-cuda]==2.15.0

# Or PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Current Status

Your system:
- ✅ RTX 3070 detected
- ✅ CUDA 13.1 installed
- ✅ NVIDIA Driver 528.92
- ❌ JAX using CPU backend (not detecting GPU)

After successful installation:
- Backend should be: `gpu` or `cuda`
- Devices should show: `[cuda(id=0)]` or `[GpuDevice(id=0)]`

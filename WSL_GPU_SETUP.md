# Complete WSL2 + JAX GPU Setup Guide

## Step-by-Step Instructions

### 1. Create Fresh Virtual Environment in WSL

```bash
# Navigate to your project
cd /mnt/e/5th\ SEM\ Data/MainEL/Final_Project/backend

# Remove any existing venv if you want fresh start
# rm -rf venv

# Create new venv (use python3.10 or python3.11 for best compatibility)
python3 -m venv venv

# Activate it
source venv/bin/activate

# Verify you're in venv (should show venv path)
which python
# Expected: /mnt/e/5th SEM Data/MainEL/Final_Project/backend/venv/bin/python
```

### 2. Install JAX with CUDA 12 Support

```bash
# Make sure venv is activated
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install JAX with CUDA 12 (this WILL have GPU support on Linux)
pip install jax[cuda12_pip]==0.4.23 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# OR try latest version:
# pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 3. Test GPU Detection

```bash
# Test 1: Check backend
python -c "import jax; print('Backend:', jax.default_backend())"
# Expected: Backend: gpu

# Test 2: List devices
python -c "import jax; print('Devices:', jax.devices())"
# Expected: Devices: [cuda(id=0)] or [GpuDevice(id=0)]

# Test 3: Run computation
python -c "import jax.numpy as jnp; import time; start=time.time(); x=jnp.ones((5000,5000)); r=jnp.dot(x,x); r.block_until_ready(); print(f'Took {time.time()-start:.3f}s')"
# Expected: < 0.5 seconds (GPU) vs 2-5 seconds (CPU)
```

### 4. If GPU Not Detected

**Check NVIDIA driver in WSL:**
```bash
nvidia-smi
```

If `nvidia-smi` doesn't work in WSL:
- Update Windows to latest version
- Update NVIDIA drivers on Windows (not inside WSL)
- Install NVIDIA CUDA Toolkit for WSL: https://docs.nvidia.com/cuda/wsl-user-guide/index.html

**Install CUDA Toolkit in WSL (if needed):**
```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA Toolkit
sudo apt-get install -y cuda-toolkit-12-3
```

### 5. Install Project Dependencies

```bash
# Activate venv
source venv/bin/activate

# Install from requirements.txt (but replace jax versions)
pip install -r requirements.txt

# Then reinstall JAX with CUDA
pip uninstall jax jaxlib -y
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 6. Run Your Test Script

```bash
# Make sure you're in WSL with venv activated
cd /mnt/e/5th\ SEM\ Data/MainEL/Final_Project/backend
source venv/bin/activate

# Run test
python test_gpu.py
```

**Expected Output:**
```
Backend: gpu  ✅
Devices: [cuda(id=0)]  ✅
GraphCast mock mode: True  (only because no model_path set)
```

## Common Issues

### Issue 1: "Defaulting to user installation"
**Problem:** venv not activated properly  
**Solution:** Run `source venv/bin/activate` first

### Issue 2: "Backend: cpu"
**Problem:** Installed CPU-only JAX  
**Solution:** 
```bash
pip uninstall jax jaxlib -y
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Issue 3: nvidia-smi doesn't work in WSL
**Problem:** NVIDIA drivers not configured for WSL  
**Solution:** 
1. Update NVIDIA drivers on Windows (must be 470.76+)
2. Update Windows to support WSL GPU (Windows 11 or Windows 10 21H2+)
3. Restart WSL: `wsl --shutdown` then relaunch

### Issue 4: "cuda12_pip" not found
**Problem:** Using old pip or wrong syntax  
**Solution:**
```bash
pip install --upgrade pip
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Verify Complete Setup

Run all these commands:

```bash
# 1. Check you're in venv
which python
# Should show: /mnt/e/.../venv/bin/python

# 2. Check nvidia-smi works
nvidia-smi
# Should show RTX 3070

# 3. Check JAX backend
python -c "import jax; print(jax.default_backend())"
# Should show: gpu

# 4. Check JAX devices
python -c "import jax; print(jax.devices())"
# Should show: [cuda(id=0)]

# 5. Run full test
python test_gpu.py
```

## Working Setup Checklist

- [ ] WSL2 installed (you have this ✅)
- [ ] Ubuntu distribution installed
- [ ] NVIDIA drivers 470.76+ on Windows
- [ ] nvidia-smi works inside WSL
- [ ] Python venv created in WSL
- [ ] JAX with cuda12_pip installed
- [ ] Backend shows 'gpu'
- [ ] Devices show 'cuda(id=0)'

## Quick Commands Reference

```bash
# Navigate to project
cd /mnt/e/5th\ SEM\ Data/MainEL/Final_Project/backend

# Activate venv
source venv/bin/activate

# Install JAX GPU
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Test GPU
python -c "import jax; print(jax.devices())"

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## Notes

- **Windows venv ≠ WSL venv**: They're separate. Create new venv in WSL.
- **JAX on Linux/WSL**: Full CUDA support ✅
- **JAX on Windows**: CPU-only ❌
- Your RTX 3070 will work perfectly in WSL2 with proper setup!

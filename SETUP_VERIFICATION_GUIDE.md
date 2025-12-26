# Setup Verification & GPU Configuration Guide

## Summary of Current Status

### ✅ What's Working:
1. **NVIDIA GPU Detected**: RTX 3070 with 8GB VRAM
2. **CUDA Toolkit**: Version 13.1 installed
3. **NVIDIA Driver**: Version 528.92

### ❌ Critical Issues:
1. **PostgreSQL Not Running**: Database connection refused
2. **JAX Not Installed in venv**: Only base Anaconda environment
3. **Using Mock Mode**: GraphCast running in mock mode (no real GPU computation)

---

## 1. Migration Verification

### Check Migration Status:

**Exit Code 0 = Success** ✅  
Your previous `python manage.py migrate` had exit code 0, which means **migrations were successful**.

However, **PostgreSQL is not currently running**, so you can't verify now. To check later:

```bash
# Activate your venv first
cd "E:\5th SEM Data\MainEL\Final_Project\backend"
.\venv\Scripts\activate

# Then check migrations
python manage.py showmigrations

# Look for [X] = applied, [ ] = not applied
# All should show [X] if migration was successful
```

**Signs of successful migration:**
- All migrations show `[X]`
- No error messages
- Exit code 0
- Can query database tables

---

## 2. Fix PostgreSQL Issue

**You need to start PostgreSQL first!**

### Option A: Windows Service
```powershell
# Check if PostgreSQL service exists
Get-Service -Name "*postgres*"

# Start PostgreSQL service
Start-Service postgresql-x64-14  # or your version name

# Or use services.msc GUI
# Press Win+R, type services.msc, find PostgreSQL, right-click Start
```

### Option B: Docker (Recommended for this project)
```bash
# Your project has docker-compose.yml
cd "E:\5th SEM Data\MainEL\Final_Project"
docker-compose up -d

# This should start PostgreSQL with PostGIS
```

---

## 3. Install JAX with GPU Support for RTX 3070

### Current Problem:
- JAX is listed in requirements.txt but NOT installed in your venv
- You're using base Anaconda environment (won't use GPU properly)
- Code is running in **mock mode** (see error: "JAX not available. GraphCast will run in mock mode")

### Solution: Install JAX with CUDA 12 support

**Step 1: Activate your venv**
```bash
cd "E:\5th SEM Data\MainEL\Final_Project\backend"
.\venv\Scripts\activate
```

**Step 2: Install JAX with CUDA 12 (for CUDA 13.1, use cuda12)**
```bash
# Remove CPU-only version if exists
pip uninstall jax jaxlib -y

# Install GPU version for CUDA 12 (compatible with CUDA 13)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify installation
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"
```

**You should see output like:**
```
JAX version: 0.4.23
Devices: [cuda(id=0)]  # ✅ This means GPU is detected!
```

### Alternative: TensorFlow GPU

If you prefer TensorFlow over JAX:

```bash
pip install tensorflow[and-cuda]

# Verify
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

---

## 4. Update Requirements.txt

Replace CPU-only JAX with GPU version:

```txt
# Replace these lines:
jax==0.4.23
jaxlib==0.4.23

# With this:
jax[cuda12_pip]==0.4.23
```

Or add at the end:
```txt
# GPU acceleration
jax[cuda12_pip]==0.4.23
tensorflow[and-cuda]==2.15.0  # Optional, if you want TensorFlow too
```

---

## 5. Verify GPU is Actually Being Used

### Test Script:
```bash
cd "E:\5th SEM Data\MainEL\Final_Project\backend"
.\venv\Scripts\activate
python verify_setup.py
```

**Expected Output:**
- ✅ All migrations applied
- ✅ GPU detected with cuda(id=0)
- ✅ JAX with CUDA support detected
- ✅ GPU computation test passed

### Real vs Mock Mode

**Mock Mode (Current - BAD):**
```
JAX not available. GraphCast will run in mock mode.
```
This means:
- No real GPU computation
- Using CPU (bottleneck)
- Predictions are simulated/fake

**Real GPU Mode (Goal - GOOD):**
```python
import jax
print(jax.devices())
# Output: [cuda(id=0)]  ✅
```

---

## 6. Complete Verification Checklist

Run these commands in order:

```bash
# 1. Start PostgreSQL (choose one method from above)
docker-compose up -d  # OR Start-Service postgresql-x64-14

# 2. Activate venv
cd "E:\5th SEM Data\MainEL\Final_Project\backend"
.\venv\Scripts\activate

# 3. Install GPU JAX
pip uninstall jax jaxlib -y
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 4. Verify migrations
python manage.py showmigrations

# 5. Verify GPU setup
python verify_setup.py

# 6. Test GPU computation
python -c "import jax; import jax.numpy as jnp; x = jnp.ones((5000, 5000)); print('Testing GPU...'); result = jnp.dot(x, x); print('✅ GPU test passed!', result.shape)"
```

---

## 7. Performance Monitoring

### Monitor GPU Usage During Execution:
```bash
# In separate terminal, watch GPU usage in real-time
nvidia-smi -l 1
```

**When GPU is working:**
- GPU Utilization: 60-100%
- Memory Usage: Increasing during computation
- Power Draw: 60-115W (not just 14W idle)

**When using CPU (bottleneck):**
- GPU Utilization: 0-5%
- Memory Usage: Minimal
- Power Draw: ~14W

---

## 8. Mock Detection in Your Code

Check these files for mock mode:

```bash
grep -r "mock mode" backend/
grep -r "JAX not available" backend/
```

**Typical pattern:**
```python
try:
    import jax
    USE_GPU = True
except ImportError:
    print("JAX not available. GraphCast will run in mock mode.")
    USE_GPU = False
```

---

## Quick Commands Reference

```bash
# Verify PostgreSQL is running
Get-Service -Name "*postgres*"

# Check if in venv (should show venv path)
python -c "import sys; print(sys.prefix)"

# Check JAX backend
python -c "import jax; print('Backend:', jax.default_backend())"
# Should output: Backend: gpu

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Monitor GPU in real-time
nvidia-smi -l 1
```

---

## Final Notes

1. **Always activate venv** before running any Python commands
2. **Don't use base Anaconda** - it won't have GPU JAX
3. **Monitor nvidia-smi** to confirm GPU usage during ML operations
4. **Exit code 0 = success** - your migration worked
5. **Mock mode = CPU bottleneck** - install GPU JAX to fix

Your RTX 3070 is ready and waiting - you just need to install the GPU-enabled JAX in your venv!

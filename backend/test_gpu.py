"""
Quick GPU Test - Run this after installing JAX with GPU support
"""

def test_gpu_availability():
    print("=" * 70)
    print("QUICK GPU TEST FOR RTX 3070")
    print("=" * 70)
    
    # Test 1: Check if JAX is installed
    print("\n1. Testing JAX Installation...")
    try:
        import jax
        print(f"   ✅ JAX version: {jax.__version__}")
    except ImportError as e:
        print(f"   ❌ JAX not installed: {e}")
        print("\n   Install with:")
        print('   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html')
        return
    
    # Test 2: Check backend
    print("\n2. Checking JAX Backend...")
    backend = jax.default_backend()
    print(f"   Backend: {backend}")
    
    if backend == 'gpu':
        print("   ✅ GPU backend active!")
    elif backend == 'cuda':
        print("   ✅ CUDA backend active!")
    else:
        print(f"   ⚠️  WARNING: Using {backend} backend (should be 'gpu' or 'cuda')")
    
    # Test 3: List devices
    print("\n3. Available Devices...")
    devices = jax.devices()
    for i, device in enumerate(devices):
        print(f"   Device {i}: {device}")
        if device.platform == 'gpu':
            print(f"      ✅ GPU device detected!")
        elif device.platform == 'cuda':
            print(f"      ✅ CUDA device detected!")
        else:
            print(f"      ⚠️  WARNING: Platform is '{device.platform}'")
    
    # Test 4: Run GPU computation
    print("\n4. Running GPU Computation Test...")
    try:
        import jax.numpy as jnp
        import time
        
        # Create large matrix
        size = 5000
        print(f"   Creating {size}x{size} matrix...")
        
        start = time.time()
        x = jnp.ones((size, size))
        result = jnp.dot(x, x)
        
        # Force computation to complete
        result.block_until_ready()
        
        elapsed = time.time() - start
        print(f"   ✅ Matrix multiplication completed in {elapsed:.3f} seconds")
        print(f"   Result shape: {result.shape}")
        print(f"   Result sum: {float(result.sum()):,.0f}")
        
        # Performance indicator
        if elapsed < 1.0:
            print("   🚀 EXCELLENT: GPU is being used!")
        elif elapsed < 5.0:
            print("   ⚡ GOOD: Likely using GPU")
        else:
            print("   ⚠️  SLOW: Might be using CPU (should be <1 second on GPU)")
        
    except Exception as e:
        print(f"   ❌ Computation test failed: {e}")
    
    # Test 5: Check GraphCast can use GPU
    print("\n5. Checking GraphCast Mock Mode Status...")
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from ml_pipeline.graphcast.inference import JAX_AVAILABLE, GraphCastInference
        
        print(f"   JAX_AVAILABLE: {JAX_AVAILABLE}")
        
        gc = GraphCastInference()
        print(f"   use_mock: {gc.use_mock}")
        
        if not gc.use_mock and JAX_AVAILABLE:
            print("   ✅ GraphCast will use GPU (mock mode disabled)")
        else:
            print("   ⚠️  GraphCast in mock mode (no real GPU computation)")
            
            if not JAX_AVAILABLE:
                print("      Reason: JAX not available")
            if gc.use_mock:
                print("      Reason: Model path not set or mock forced")
                
    except Exception as e:
        print(f"   ⚠️  Could not check GraphCast: {e}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    # Summary
    print("\n📊 SUMMARY:")
    if backend in ['gpu', 'cuda']:
        print("   ✅ JAX is configured for GPU")
        print("   ✅ Your RTX 3070 should be utilized")
        print("\n   To monitor GPU usage during ML operations:")
        print("   nvidia-smi -l 1")
    else:
        print("   ❌ JAX is using CPU - GPU not configured")
        print("\n   Fix by installing JAX with CUDA support:")
        print('   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html')

if __name__ == "__main__":
    test_gpu_availability()

"""
Verification script for Django migrations and GPU/CUDA setup
"""
import sys
import subprocess

def check_migration_status():
    """Check if Django migrations were successful"""
    print("=" * 60)
    print("CHECKING DJANGO MIGRATIONS")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ['python', 'manage.py', 'showmigrations'],
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stdout
        print(output)
        
        # Check for unapplied migrations
        if '[ ]' in output:
            print("\n⚠️  WARNING: Some migrations are NOT applied!")
            return False
        else:
            print("\n✅ All migrations are applied successfully!")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error checking migrations: {e}")
        return False

def check_cuda_setup():
    """Check CUDA and GPU availability"""
    print("\n" + "=" * 60)
    print("CHECKING CUDA & GPU SETUP")
    print("=" * 60)
    
    # Check JAX
    print("\n1. Checking JAX GPU support...")
    try:
        import jax
        print(f"   JAX version: {jax.__version__}")
        
        devices = jax.devices()
        print(f"   Available devices: {devices}")
        
        # Check if GPU is available
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        if gpu_devices:
            print(f"   ✅ GPU detected: {gpu_devices}")
            
            # Test GPU computation
            import jax.numpy as jnp
            x = jnp.ones((1000, 1000))
            result = jnp.dot(x, x)
            print(f"   ✅ GPU computation test passed")
            
        else:
            print(f"   ⚠️  WARNING: No GPU detected! JAX is using CPU")
            print(f"   Current JAX platform: {jax.default_backend()}")
            
    except ImportError:
        print("   ❌ JAX is not installed")
    except Exception as e:
        print(f"   ⚠️  JAX error: {e}")
    
    # Check NVIDIA GPU
    print("\n2. Checking NVIDIA GPU...")
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        print("   ✅ NVIDIA GPU detected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ❌ nvidia-smi not found or failed")
    
    # Check CUDA availability
    print("\n3. Checking CUDA version...")
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        print("   ✅ CUDA toolkit detected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ⚠️  nvcc not found (CUDA toolkit may not be installed)")
    
    # Check TensorFlow (if installed)
    print("\n4. Checking TensorFlow GPU support...")
    try:
        import tensorflow as tf
        print(f"   TensorFlow version: {tf.__version__}")
        print(f"   GPU available: {tf.config.list_physical_devices('GPU')}")
        
        if tf.config.list_physical_devices('GPU'):
            print("   ✅ TensorFlow GPU support enabled")
        else:
            print("   ⚠️  WARNING: TensorFlow not detecting GPU")
            
    except ImportError:
        print("   ℹ️  TensorFlow is not installed")
    except Exception as e:
        print(f"   ⚠️  TensorFlow error: {e}")

def check_jax_installation():
    """Check if JAX is CPU or GPU version"""
    print("\n" + "=" * 60)
    print("JAX INSTALLATION ANALYSIS")
    print("=" * 60)
    
    try:
        import jaxlib
        print(f"JAXlib version: {jaxlib.__version__}")
        
        # Check if it's CUDA version
        if 'cuda' in jaxlib.__version__:
            print("✅ JAX with CUDA support detected")
        else:
            print("⚠️  WARNING: JAX appears to be CPU-only version")
            print("\nTo install JAX with CUDA 12 support for RTX 3070:")
            print("   pip install --upgrade 'jax[cuda12_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
            print("\nOr for CUDA 11:")
            print("   pip install --upgrade 'jax[cuda11_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
            
    except ImportError:
        print("❌ JAXlib not found")

if __name__ == "__main__":
    # Check migrations
    migration_ok = check_migration_status()
    
    # Check GPU/CUDA setup
    check_cuda_setup()
    
    # Check JAX installation
    check_jax_installation()
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

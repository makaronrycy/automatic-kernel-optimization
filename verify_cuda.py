"""
Verify CUDA and PyTorch installation
"""
import sys

def verify_installation():
    print("=" * 80)
    print("PyTorch CUDA Verification")
    print("=" * 80)
    
    try:
        import torch
        print(f"\n✓ PyTorch installed successfully")
        print(f"  Version: {torch.__version__}")
    except ImportError as e:
        print(f"\n✗ PyTorch not installed: {e}")
        return False
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"  Device Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Compute Capability: {torch.cuda.get_device_capability(i)}")
            print(f"    Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # Test tensor creation on CUDA
        try:
            test_tensor = torch.randn(100, 100).cuda()
            print(f"\n✓ Successfully created tensor on CUDA")
            print(f"  Tensor device: {test_tensor.device}")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n✗ Failed to create tensor on CUDA: {e}")
            return False
    else:
        print("\n⚠ CUDA not available - running on CPU only")
        print("\nPossible reasons:")
        print("  1. NVIDIA GPU drivers not installed")
        print("  2. CUDA toolkit not installed")
        print("  3. PyTorch installed without CUDA support")
        print("\nTo install PyTorch with CUDA support:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    # Check other dependencies
    print("\n" + "=" * 80)
    print("Other Dependencies")
    print("=" * 80)
    
    try:
        import numpy
        print(f"✓ NumPy: {numpy.__version__}")
    except ImportError:
        print("✗ NumPy not installed")
    
    try:
        import matplotlib
        print(f"✓ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("✗ Matplotlib not installed")
    
    try:
        import psutil
        print(f"✓ psutil: {psutil.__version__}")
    except ImportError:
        print("✗ psutil not installed")
    
    print("\n" + "=" * 80)
    
    return cuda_available


if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)

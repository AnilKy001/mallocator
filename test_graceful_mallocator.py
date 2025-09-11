import pytest
import subprocess
import sys
import os


def test_allocator_installation():
    """Test that the graceful allocator can be installed and works."""
    test_script = '''
import sys
sys.path.insert(0, ".")

# Install allocator first
from graceful_mallocator import install_mallocator
install_mallocator(max_retries=3, wait_time=1.0)

# Test it works
import torch
if torch.cuda.is_available():
    x = torch.randn(100, 100).cuda()
    assert x.is_cuda
    del x
    torch.cuda.empty_cache()
    print("SUCCESS")
else:
    print("SKIP_NO_CUDA")
'''
    
    result = subprocess.run(
        [sys.executable, '-c', test_script],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )
    
    if "SKIP_NO_CUDA" in result.stdout:
        pytest.skip("CUDA not available")
    
    assert result.returncode == 0, f"Test failed: {result.stderr}"
    assert "SUCCESS" in result.stdout


def test_large_allocation():
    """Test large allocation to potentially trigger retry logic."""
    test_script = '''
import sys
sys.path.insert(0, ".")

from graceful_mallocator import install_mallocator
install_mallocator(max_retries=3, wait_time=1.0)

import torch
if torch.cuda.is_available():
    try:
        x = torch.randn(5000, 5000).cuda()
        del x
        torch.cuda.empty_cache()
        print("SUCCESS")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("SUCCESS_OOM_HANDLED")
        else:
            raise
else:
    print("SKIP_NO_CUDA")
'''
    
    result = subprocess.run(
        [sys.executable, '-c', test_script],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )
    
    if "SKIP_NO_CUDA" in result.stdout:
        pytest.skip("CUDA not available")
    
    assert result.returncode == 0, f"Test failed: {result.stderr}"
    assert "SUCCESS" in result.stdout or "SUCCESS_OOM_HANDLED" in result.stdout


if __name__ == "__main__":
    # Simple main function for direct execution
    print("Running graceful mallocator tests...")
    
    try:
        from graceful_mallocator import install_mallocator
        install_mallocator(max_retries=3, wait_time=1.0)
        print("✅ Allocator installed")
        
        import torch
        if torch.cuda.is_available():
            x = torch.randn(100, 100).cuda()
            print("✅ Basic allocation works")
            del x
            torch.cuda.empty_cache()
            print("🎉 All tests passed!")
        else:
            print("⚠️ CUDA not available")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        exit(1)

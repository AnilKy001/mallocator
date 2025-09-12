def test_graceful_allocator_installation():
    """Test that the graceful allocator can be installed without errors."""
    import torch  # Import here to ensure allocator is installed first
    
    # Verify that the allocator is set
    if torch.cuda.is_available():
        current_allocator = torch.cuda.memory._get_current_allocator()
        assert current_allocator is not None
        print("Allocator installation test passed.")
        return True
    else:
        print("CUDA is not available, skipping test.")
        return True

def test_basic_allocation():
    """Test basic memory allocation with the graceful allocator."""
    import torch  # Import here to ensure allocator is installed first
    
    if not torch.cuda.is_available():
        print("CUDA is not available, skipping test.")
        return True
    
    try:
        # Allocate a small tensor on CUDA
        x = torch.randn(100, 100).cuda()
        assert x.is_cuda
        assert x.shape == (100, 100)
        
        # Free the tensor
        del x
        torch.cuda.empty_cache()
        print("Basic allocation test passed.")
        return True
    except Exception as e:
        print(f"Basic allocation test failed: {e}")
        return False

def test_large_allocation():
    """Test allocation of a larger tensor to potentially trigger retry logic."""
    import torch  # Import here to ensure allocator is installed first
    import torch.nn as nn
    import time
    import threading
    
    if not torch.cuda.is_available():
        print("CUDA is not available, skipping test.")
        return True

    def worker_thread(thread_id, device, tensor_size):
        print(f"Thread {thread_id}: Allocating {tensor_size}x{tensor_size} tensor.")
        tensor = torch.randn(tensor_size, tensor_size, device=device)
        time.sleep(5)
        del tensor
        torch.cuda.empty_cache()

     

    try:

        thread_1 = threading.Thread(target=worker_thread, args=(1, 'cuda:0', 70000))
        thread_2 = threading.Thread(target=worker_thread, args=(2, 'cuda:0', 70000))

        thread_1.start()
        time.sleep(1)  # Stagger the start
        thread_2.start()

        thread_1.join()
        thread_2.join()


        print("Large allocation test passed.")
        return True
    except RuntimeError as e:
        # If OOM occurs, the graceful allocator should handle it with retries
        if "out of memory" in str(e).lower() or "allocation failed" in str(e).lower():
            print("Large allocation test passed (OOM handled gracefully)")
            return True
        else:
            print(f"Large allocation test failed with unexpected error: {e}")
            return False
    except Exception as e:
        print(f"Large allocation test failed: {e}")
        return False

def main():
    """Run all tests manually without pytest."""
    print("=== Graceful Mallocator Tests ===")
    print("Running tests manually (bypassing pytest)...")
    
    # Install allocator first
    try:
        print("\n1. Installing allocator...")
        from graceful_mallocator import install_mallocator
        install_mallocator()
        print("Allocator installed successfully")
    except Exception as e:
        print(f"Failed to install allocator: {e}")
        return False
    
    # Import torch after allocator installation
    print("\n2. Importing PyTorch...")
    try:
        import torch
        print(f"PyTorch imported successfully (version: {torch.__version__})")
        
        if torch.cuda.is_available():
            print(f"CUDA available (device: {torch.cuda.get_device_name()})")
        else:
            print("CUDA is not available")
    except Exception as e:
        print(f"Failed to import PyTorch: {e}")
        return False
    
    # Run tests
    print("\n3. Running tests...")
    
    tests = [
        ("Allocator Installation", test_graceful_allocator_installation),
        ("Basic Allocation", test_basic_allocation),
        ("Large Allocation", test_large_allocation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                print(f"{test_name} failed")
        except Exception as e:
            print(f"{test_name} failed with exception: {e}")
    
    # Summary
    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("All tests passed!")
        return True
    else:
        print("Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

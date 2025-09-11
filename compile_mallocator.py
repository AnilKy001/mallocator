#!/usr/bin/env python3
"""
Pre-compile the graceful mallocator extension.
Run this ONCE to compile the extension, then use it without recompilation.
"""

import torch
import torch.utils.cpp_extension
import os
import hashlib

def compile_graceful_mallocator(max_retries=5, wait_time=5.0):
    """Compile the graceful mallocator extension ahead of time."""
    
    mallocator_source = """
#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include <fstream>
#include <cstdlib>

extern "C" {{

bool check_memory_available(size_t bytes_required, int device) {{
    size_t bytes_free, bytes_total;
    // Set device only when this function is actually called, not during module load
    cudaError_t set_err = cudaSetDevice(device);
    if (set_err != cudaSuccess) {{
        return false;
    }}
    cudaError_t err = cudaMemGetInfo(&bytes_free, &bytes_total);
    if ((err == cudaSuccess) && (bytes_free >= bytes_required)) {{
        return true;
    }} 
    return false;
}}

void signal_oom_external (size_t size, int device) {{
    // Placeholder for signaling logic
    return;
}}

void* graceful_malloc (size_t size, int device, cudaStream_t stream) {{
    for (int attempt = 1; attempt <= {max_retries}; attempt++) {{
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);

        if (err == cudaSuccess) {{
            return ptr;
        }}
        else if (err == cudaErrorMemoryAllocation) {{
            std::cerr << "Out-of-memory error has occurred during the allocation of " << size << " bytes on device " << device << ". Attempt " << attempt << " of {max_retries}." << std::endl;
            if (attempt == 1) {{
                signal_oom_external(size, device);
            }}
            if (attempt < {max_retries}) {{
                std::cout << "Retrying after waiting {wait_time} seconds..." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(int({wait_time} * 1000)));

                // Synchronize to ensure all operations are complete
                cudaDeviceSynchronize();

                // Wait until sufficient memory is available, then try allocation again:
                std::cout << "Checking for available memory..." << std::endl;
                bool memory_available = false;
                for (int check_count = 0; check_count < 5; check_count++) {{
                    if (check_memory_available(size, device)) {{
                        memory_available = true;
                        std::cout << "Sufficient memory available. Retrying allocation." << std::endl;
                        break;
                    }}
                    else {{
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    }}
                }}
                if (memory_available == false) {{
                    std::cout << "Memory allocation failed for attempt " << attempt << "." << std::endl;
                }}
            }}
            else {{
                std::cerr << "Max retries reached ({max_retries}). Allocation failed." << std::endl;
                return nullptr;
            }}
        }}
        else {{
            std::cerr << "Non-OOM error during allocation: " << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }}
    }}
    return nullptr;
}}

void graceful_free (void* ptr, size_t size, int device, cudaStream_t stream) {{
    if (ptr) {{
        cudaFree(ptr);
    }}
    return;
}}

}}

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{}}
""".format(max_retries=max_retries, wait_time=wait_time)

    print("Compiling graceful mallocator extension...")
    print("(This will initialize CUDA, but that's OK since we're pre-compiling)")
    
    build_dir = os.path.abspath("./build")
    os.makedirs(build_dir, exist_ok=True)
    
    extension_name = f"graceful_mallocator"
    
    try:
        extension_path = torch.utils.cpp_extension.load_inline(
            name=extension_name,
            cpp_sources=mallocator_source,
            with_cuda=True,
            verbose=True,
            is_python_module=False,
            build_directory=build_dir
        )
        
        print(f"✅ Extension compiled successfully!")
        print(f"   Location: {extension_path}")
        print()
        print("Now you can use the allocator without recompilation:")
        print("   from graceful_mallocator import install_mallocator")
        print("   install_mallocator()")
        
        return extension_path
        
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        raise

if __name__ == "__main__":
    print("Graceful Mallocator Pre-Compiler")
    print("=" * 40)
    print("This script compiles the CUDA extension ahead of time.")
    print("Run this once, then use the compiled extension without")
    print("triggering CUDA initialization during import.")
    print()
    
    try:
        compile_graceful_mallocator()
    except Exception as e:
        print(f"Failed to compile: {e}")
        exit(1)

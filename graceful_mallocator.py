import os
import torch
import torch.utils.cpp_extension
from torch.cuda.memory import CUDAPluggableAllocator

def install_mallocator(
        max_retries: int = 5,
        wait_time: float = 5.0,
        signal_on_oom: bool = False
):
    mallocator_source = """
#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include <fstream>
#include <cstdlib>

extern "C" {

bool check_memory_available(size_t bytes_required, int device) {{
    size_t bytes_free, bytes_total;
    cudaSetDevice(device);
    cudaError_t err = cudaMemGetInfo(&bytes_free, &bytes_total);
    if ((err == cudaSuccess) && (bytes_free >= bytes_required)) {{
        return true;
    }} 
    return false;
}}

void signal_oom_external (size_t size, int device) {
    // Placeholder for signaling logic
    return;
}

void* graceful_malloc (size_t size, int device, cudaStream_t stream) {{
    for (int attempt = 0; attempt <= {max_retries}; ++attempt) {{
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);

        if (err == cudaSuccess) {{
            return ptr;
        }}
        else if (err == cudaErrorMemoryAllocation) {{
            std::cerr << "Out-of-memory error has occurred during the allocation of " << size << " bytes on device " << device << ". Attempt " << attempt << " of {max_retries}." << std::endl;
            if (attempt == 0) {{
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
                for (int check_count = 0; check < 5; check_count++) {{
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
                    ctd::cout << "Memory allocation failed for attempt " << attempt << "." << std::endl;
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
    std::cerr << "Maximum retries reached. Allocation failed after {max_retries} retries." << std::endl;
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
"""

    graceful_mallocator_module = torch.utils.cpp_extension.load_inline(
        name="graceful_mallocator",
        cpp_sources=mallocator_source,
        with_cuda=True,
        verbose=False,
        is_python_module=False,
        build_directory="./"
    )

    graceful_mallocator = CUDAPluggableAllocator(
        graceful_mallocator_module.__file__,
        "graceful_malloc",
        "graceful_free"
    )

    torch.cuda.memory.change_current_allocator(graceful_mallocator)

    print("Installed graceful memory allocator with the following parameters:")
    print(f"  Maximum number of retries: {max_retries}")
    print(f"  Wait time between retries: {wait_time} seconds")
    print(f"  Signal on OOM: {'Enabled' if signal_on_oom else 'Disabled'}")
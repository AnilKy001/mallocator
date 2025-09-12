import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
import torch
import torch.utils.cpp_extension
from torch.cuda.memory import CUDAPluggableAllocator

def install_mallocator():

    graceful_mallocator = torch.cuda.memory.CUDAPluggableAllocator(
        "./build/graceful_mallocator.so",
        "graceful_malloc",
        "graceful_free"
    )

    torch.cuda.memory.change_current_allocator(graceful_mallocator)

    print("Installed graceful memory allocator.")
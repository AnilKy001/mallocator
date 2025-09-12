# Graceful Mallocator

A PyTorch CUDA memory allocator that handles out-of-memory (OOM) errors gracefully by waiting and retrying allocation instead of crashing.

## What It Does

- **Prevents crashes**: When GPU memory runs out, waits and retries instead of terminating
- **Configurable retry delay**: Set custom wait time before retry attempts (default: 5 seconds)
- **Drop-in replacement**: No code changes needed in your PyTorch applications
- **Memory monitoring**: Continuously checks for available memory before retrying

## Installation & Usage

### 1. Compile the allocator (run once)

```bash
# Default 5-second wait time
python compile_mallocator.py

# Custom wait time (e.g., 10 seconds)
python compile_mallocator.py --wait-time 10.0
```

### 2. Use in your PyTorch code

```python
from graceful_mallocator import install_mallocator
import torch

# Install before any CUDA operations
install_mallocator()

# Use PyTorch normally - OOM errors now handled gracefully
x = torch.randn(10000, 10000).cuda()
model = torch.nn.Linear(1000, 1000).cuda()
```

## Requirements

- PyTorch with CUDA support
- CUDA toolkit
- C++ compiler

## Testing

Run the included tests to verify functionality:

```bash
python test_graceful_mallocator.py
```

The tests cover basic allocation, large tensor allocation, and multi-threaded scenarios that may trigger OOM conditions.

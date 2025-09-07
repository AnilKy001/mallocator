import torch
import time
import logging
import os
import signal
import threading
from typing import Optional, Callable, Dict, Any
from functools import wraps

class torch_mallocator:
    def __init__(
            self,
            delay: float = 1.0,
            max_retries: int = 10,
            signal_on_oom: bool = False,
            oom_signal: int = signal.SIGUSR1
    ):
        self.delay = delay
        self.max_retries = max_retries
        self.signal_on_oom = signal_on_oom
        self.oom_signal = oom_signal

        self.stats = {
            'num_allocations': 0,
            'num_oom_events': 0,
            'num_successful_retries': 0,
            'num_failed_allocations': 0,
            'total_allocated_bytes': 0
        }

        self.thread_local = threading.local()

        self.torch_vanilla_malloc = None
        self.torch_vanilla_delete = None
        self.mallocator_installed = False

    def get_thread_local_retry_count(self):
        if not hasattr(self.thread_local, 'retry_count'):
            self.thread_local.retry_count = 0
        return self.thread_local.retry_count
    
    def set_thread_local_retry_count(self, count):
        self.thread_local.retry_count = count

    def signal_oom_event(self):
        return

    def custom_malloc(
            self,
            size: int,
            device: torch.device,
            stream: Optional[torch.cuda.Stream] = None
    ):
        '''
        Wraps the original PyTorch CUDA allocation
        '''
        self.stats['num_allocations'] += 1
        print(f"Attempting to allocate {size} bytes on {device}")

        retry_count = self.get_thread_local_retry_count()

        while retry_count < self.max_retries:
            try:
                self.torch_vanilla_malloc(size, device, stream)
                self.stats['total_allocated_bytes'] += size
            except RuntimeError as e:
                print(f"OOM encountered during allocation of {size} bytes on {device}: {e}")
                self.stats['num_oom_events'] += 1
                retry_count += 1
                self.set_thread_local_retry_count(retry_count)
                self.signal_oom_event()

                if retry_count >= self.max_retries:
                    print(f"Max retries reached ({self.max_retries}). Allocation failed.")
                    self.stats['num_failed_allocations'] += 1
                    raise e
                else:
                    time.sleep(self.delay)
                    torch.cuda.memory.empty_cache()
        
        self.set_thread_local_retry_count(0)
        self.stats['num_successful_retries'] += retry_count
        print(f"Allocation of {size} bytes on {device} succeeded after {retry_count} retries.")
        return
    

                    







    



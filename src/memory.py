"""Thin wrappers around CUDA memory APIs used for the memory benchmark.

Workflow: call ``reset_peak()`` before a measured interval, then read
``max_allocated_mb()`` afterwards for the exact peak observed during it.
"""
import torch

_BYTES_PER_MB = 1024 * 1024


def reset_peak(device=None) -> None:
    """Reset the peak-memory counter so the next read covers a fresh interval."""
    torch.cuda.reset_peak_memory_stats(device)


def max_allocated_mb(device=None) -> float:
    """Peak bytes allocated since the last reset, in MiB."""
    return torch.cuda.max_memory_allocated(device) / _BYTES_PER_MB


def current_allocated_mb(device=None) -> float:
    """Currently allocated bytes, in MiB."""
    return torch.cuda.memory_allocated(device) / _BYTES_PER_MB


def empty_cache() -> None:
    """Release cached-but-unused blocks back to the CUDA allocator."""
    torch.cuda.empty_cache()

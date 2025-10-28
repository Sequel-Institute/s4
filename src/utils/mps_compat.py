"""MPS (Metal Performance Shaders) compatibility utilities for Apple Silicon.

MPS on Apple Silicon doesn't support complex number operations like matmul, bmm,
and einsum. This module provides drop-in replacements that automatically fall back
to CPU for complex operations while maintaining GPU acceleration for real-valued ops.

Usage:
    from src.utils.mps_compat import mps_matmul, mps_einsum

    # Use instead of @ or torch.matmul
    result = mps_matmul(A, B)

    # Use instead of torch.einsum
    result = mps_einsum('ij,jk->ik', A, B)
"""

import torch
from typing import Any


def _needs_cpu_fallback(device: torch.device, *tensors: torch.Tensor) -> bool:
    """Check if tensors need CPU fallback due to MPS complex number limitations.

    Args:
        device: The device to check
        *tensors: Tensors to check for complex dtype

    Returns:
        True if MPS + complex numbers detected, False otherwise
    """
    if device.type != 'mps':
        return False

    for tensor in tensors:
        if tensor.dtype in [torch.complex64, torch.complex128]:
            return True

    return False


def mps_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """MPS-compatible matrix multiplication.

    Automatically falls back to CPU for complex operations on MPS,
    otherwise uses native matmul for maximum performance.

    Args:
        a: First tensor
        b: Second tensor

    Returns:
        Result of matrix multiplication on original device
    """
    if _needs_cpu_fallback(a.device, a, b):
        device = a.device
        result = a.cpu() @ b.cpu()
        return result.to(device)
    return a @ b


def mps_einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    """MPS-compatible Einstein summation.

    Automatically falls back to CPU for complex operations on MPS,
    otherwise uses native einsum for maximum performance.

    Args:
        equation: Einstein summation equation string
        *operands: Input tensors

    Returns:
        Result of einsum on original device
    """
    if not operands:
        raise ValueError("No operands provided to mps_einsum")

    if _needs_cpu_fallback(operands[0].device, *operands):
        device = operands[0].device
        cpu_operands = [op.cpu() for op in operands]
        result = torch.einsum(equation, *cpu_operands)
        return result.to(device)

    return torch.einsum(equation, *operands)


def mps_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """MPS-compatible batch matrix multiplication.

    Automatically falls back to CPU for complex operations on MPS,
    otherwise uses native bmm for maximum performance.

    Args:
        a: First tensor (batch of matrices)
        b: Second tensor (batch of matrices)

    Returns:
        Result of batch matrix multiplication on original device
    """
    if _needs_cpu_fallback(a.device, a, b):
        device = a.device
        result = torch.bmm(a.cpu(), b.cpu())
        return result.to(device)
    return torch.bmm(a, b)


def mps_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """MPS-compatible matrix-matrix product.

    Automatically falls back to CPU for complex operations on MPS,
    otherwise uses native mm for maximum performance.

    Args:
        a: First matrix
        b: Second matrix

    Returns:
        Result of matrix multiplication on original device
    """
    if _needs_cpu_fallback(a.device, a, b):
        device = a.device
        result = torch.mm(a.cpu(), b.cpu())
        return result.to(device)
    return torch.mm(a, b)


# Create an alias for backwards compatibility and convenience
contract = mps_einsum

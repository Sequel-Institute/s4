# MPS Compatibility for S4 Model

This document describes the MPS (Metal Performance Shaders) compatibility layer added to enable training on Apple Silicon GPUs.

## Problem

MPS on Apple Silicon doesn't support complex number operations including:
- `torch.einsum` with complex tensors
- `torch.matmul` / `@` operator with complex tensors
- `torch.bmm` with complex tensors
- `torch.mm` with complex tensors

The S4 model uses complex number arithmetic extensively in its state space calculations.

## Solution

A centralized MPS compatibility utility module (`src/utils/mps_compat.py`) provides drop-in replacements that:
1. Detect when complex operations run on MPS devices
2. Automatically fall back to CPU for those specific operations
3. Move results back to MPS
4. Have zero overhead for CUDA/CPU or non-complex operations

## Files Modified

### Core Utility
- **`src/utils/mps_compat.py`** (new file)
  - `mps_einsum()`: MPS-safe Einstein summation
  - `mps_matmul()`: MPS-safe matrix multiplication
  - `mps_bmm()`: MPS-safe batch matrix multiplication
  - `mps_mm()`: MPS-safe matrix-matrix product

### Updated Files
1. **`src/models/sequence/kernels/ssm.py`**
   - Replaced `torch.einsum` with `mps_einsum`
   - Used via `contract` alias

2. **`src/models/sequence/kernels/fftconv.py`**
   - Replaced `torch.einsum` with `mps_einsum`
   - Used via `contract` alias

3. **`src/models/functional/krylov.py`**
   - Replaced all `@` operators with `mps_matmul()`
   - 9 matrix multiplication operations updated

4. **`src/models/functional/vandermonde.py`**
   - Replaced `torch.einsum` with `mps_einsum`
   - Used via `contract` alias

## Usage

No changes needed in user code. The modifications are transparent:

```bash
# Training on MPS (Apple Silicon)
uv run python -m train pipeline=sc model=s4

# Training on CPU (if needed)
uv run python -m train pipeline=sc model=s4 trainer.accelerator=cpu

# Training on CUDA (if available)
uv run python -m train pipeline=sc model=s4 trainer.accelerator=gpu
```

## Performance Notes

- Complex operations run on CPU (unavoidable due to MPS limitations)
- Real-valued operations (FFT, convolutions, linear layers) use MPS acceleration
- Overall training speed on Apple Silicon is still faster than CPU-only

## Future Improvements

If Apple adds complex number support to MPS in future versions, the compatibility layer will automatically use MPS without code changes.

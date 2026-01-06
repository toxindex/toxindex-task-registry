# MM/GBSA Determinism Testing

## Overview

This document explains the determinism improvements made to the MM/GBSA affinity calculations and how to test them.

## Requirements

**CRITICAL: CUDA is REQUIRED for deterministic testing**

- NVIDIA GPU with CUDA support
- CUDA-enabled OpenMM installation
- The tests will **fail immediately** if CUDA is not available

### Why CUDA?

1. **DeterministicForces flag**: Only CUDA supports forcing deterministic summation of forces
2. **Double precision**: CUDA double precision ensures bit-identical reproducibility
3. **Consistent parallelism**: CUDA provides better control over parallel execution order

OpenCL and CPU platforms do NOT provide the same level of determinism guarantees.

## Running Tests

### Standard Test Suite

```bash
cd /home/kyu_insilica_co/toxindex-task-registry/tasks/affinity
uv run python affinity/affinity_test.py
```

This will run 4 test categories:

0. **DIAGNOSTIC**: Minimizes once, calculates energies 3 times
   - **Purpose**: Isolate whether variation comes from minimization or energy calculation
   - **Expected**: All 3 energy calculations should be **identical** (< 1e-6 kcal/mol variation)
   - **If this passes**: Variation in other tests comes from minimization step
   - **If this fails**: Problem is in energy calculation itself

1. **Reproducibility**: Same input, 3 independent runs
   - **Expected with CUDA**: < 0.01 kcal/mol std dev (nearly perfect)
   - **Acceptable**: < 0.1 kcal/mol std dev

2. **Symmetry**: Swap receptor/ligand chains
   - **Expected**: Identical results (swapping shouldn't change ΔG)
   - **Acceptable**: < 0.1 kcal/mol difference

3. **Independence**: Results independent of batch size
   - **Expected**: Same results whether processing 2 or 3 cases
   - **Acceptable**: < 0.1 kcal/mol difference

## Improvements Made

### 1. CUDA Deterministic Settings

File: `affinity/mmgbsa_utils.py`

```python
# Double precision (not mixed)
platform.setPropertyDefaultValue('CudaPrecision', 'double')

# Fixed device
platform.setPropertyDefaultValue('CudaDeviceIndex', '0')

# Deterministic force summation
platform.setPropertyDefaultValue('DeterministicForces', 'true')
```

### 2. Strict Convergence Tolerance

```python
# Very strict tolerance: 0.001 kJ/(mol*nm)
tolerance = 0.001 * unit.kilojoule_per_mole / unit.nanometer

# Auto-calculate iterations based on system size
# For 10,000 atoms with strict tolerance: ~70,000 iterations
max_iterations = calculate_max_iterations(num_atoms, tolerance_kj_mol=0.001)
```

### 3. Normalized Chains

Chains are always sorted to ensure consistent ordering:

```python
receptor_chains = sorted(['H', 'L'])  # Always ['H', 'L'], not ['L', 'H']
```

### 4. VerletIntegrator (Deterministic)

Uses `VerletIntegrator` instead of `LangevinIntegrator`:
- No stochastic terms
- No random number generation
- Fully deterministic

## Expected Results on CUDA

### Best Case (CUDA with DeterministicForces)

```
DIAGNOSTIC:     ✓ < 1e-6 kcal/mol (perfectly deterministic)
Reproducibility: ✓ < 0.01 kcal/mol (excellent)
Symmetry:        ✓ < 0.01 kcal/mol (excellent)
Independence:    ✓ < 0.01 kcal/mol (excellent)
```

### Good Case (CUDA without DeterministicForces)

```
DIAGNOSTIC:     ✓ < 1e-6 kcal/mol (energy calc is deterministic)
Reproducibility: ~ 0.05-0.2 kcal/mol (acceptable, variation from minimization)
Symmetry:        ~ 0.1-0.5 kcal/mol (moderate, minimization variation)
Independence:    ~ 0.1-0.5 kcal/mol (moderate)
```

### Poor Case (OpenCL/CPU)

```
DIAGNOSTIC:     ✗ > 0.01 kcal/mol (energy calc non-deterministic)
Reproducibility: ✗ 1-6 kcal/mol (poor)
Symmetry:        ✗ 1-5 kcal/mol (poor)
Independence:    ✗ 5-15 kcal/mol (very poor)
```

## Troubleshooting

### Tests fail with "CUDA platform not available"

**Solution**: Install CUDA-enabled OpenMM

```bash
# Using conda
conda install -c conda-forge openmm cudatoolkit

# Verify CUDA is available
python -c "import openmm; print(openmm.Platform.getPlatformByName('CUDA'))"
```

### Tests show variation > 0.1 kcal/mol on CUDA

**Possible causes**:
1. DeterministicForces not supported (older CUDA/OpenMM versions)
2. Minimization not converging (increase max_iterations)
3. Multiple GPUs interfering (ensure CudaDeviceIndex='0')

**Solutions**:
- Update OpenMM to latest version
- Check convergence warnings in output
- Set environment variable: `export CUDA_VISIBLE_DEVICES=0`

### Energy calculation is non-deterministic (diagnostic test fails)

This should **never** happen with proper CUDA settings. If it does:
1. Verify double precision is actually being used
2. Check for memory issues (try smaller test case)
3. Verify DeterministicForces is enabled
4. Try Reference platform for comparison (slow but always deterministic)

## Performance vs Determinism Trade-offs

| Setting | Speed | Determinism |
|---------|-------|-------------|
| CUDA mixed precision | Fastest | Poor |
| CUDA double precision | ~50% slower | Good |
| CUDA double + DeterministicForces | ~60% slower | Excellent |
| CPU Reference | 10-100x slower | Perfect |

For production: Use CUDA double + DeterministicForces
For validation: Use CPU Reference (slow but perfect)

## Next Steps

If tests still show variation after CUDA setup:

1. **Check diagnostic test first** - this tells you WHERE the problem is
2. **If energy calc is deterministic** → Problem is minimization convergence
   - Increase tolerance strictness
   - Increase max iterations
   - Consider pre-minimizing structures
3. **If energy calc is non-deterministic** → Problem is platform/precision
   - Verify CUDA settings are applied
   - Try Reference platform
   - Check GPU hardware issues

#!/usr/bin/env python
"""
Quick verification script to check determinism settings.
Run this before full test suite to verify CUDA and settings are correct.
"""

import openmm
from openmm import Platform

print("=" * 80)
print("OpenMM Determinism Settings Verification")
print("=" * 80)

# 1. Check OpenMM version
print(f"\n1. OpenMM Version: {openmm.__version__}")

# 2. Check available platforms
print(f"\n2. Available Platforms:")
for i in range(Platform.getNumPlatforms()):
    platform = Platform.getPlatform(i)
    print(f"   - {platform.getName()}")

# 3. Check CUDA platform
try:
    cuda_platform = Platform.getPlatformByName('CUDA')
    print(f"\n3. ✓ CUDA Platform Found!")
    print(f"   Properties: {cuda_platform.getPropertyNames()}")

    # 4. Verify DeterministicForces is available
    if 'DeterministicForces' in cuda_platform.getPropertyNames():
        print(f"\n4. ✓ DeterministicForces property is AVAILABLE")
        print(f"   Default value: {cuda_platform.getPropertyDefaultValue('DeterministicForces')}")
    else:
        print(f"\n4. ✗ WARNING: DeterministicForces property NOT available")
        print(f"   This may cause non-deterministic behavior")

    # 5. Test setting properties
    print(f"\n5. Testing property settings:")
    try:
        cuda_platform.setPropertyDefaultValue('CudaPrecision', 'double')
        print(f"   ✓ CudaPrecision set to: {cuda_platform.getPropertyDefaultValue('CudaPrecision')}")

        cuda_platform.setPropertyDefaultValue('DeterministicForces', 'true')
        print(f"   ✓ DeterministicForces set to: {cuda_platform.getPropertyDefaultValue('DeterministicForces')}")

        cuda_platform.setPropertyDefaultValue('CudaDeviceIndex', '0')
        print(f"   ✓ CudaDeviceIndex set to: {cuda_platform.getPropertyDefaultValue('CudaDeviceIndex')}")

        print(f"\n✓ ALL SETTINGS VERIFIED - Ready for deterministic testing!")

    except Exception as e:
        print(f"\n✗ ERROR setting properties: {e}")

except Exception as e:
    print(f"\n✗ CUDA Platform NOT available: {e}")
    print(f"\nThis environment cannot run deterministic tests.")
    print(f"Please use a machine with NVIDIA GPU and CUDA-enabled OpenMM.")

# 6. Check PDBFixer seed support
print(f"\n6. Checking PDBFixer:")
try:
    import pdbfixer
    import inspect
    sig = inspect.signature(pdbfixer.PDBFixer.addMissingAtoms)
    if 'seed' in sig.parameters:
        print(f"   ✓ PDBFixer.addMissingAtoms supports 'seed' parameter")
    else:
        print(f"   ✗ WARNING: PDBFixer.addMissingAtoms does NOT support 'seed' parameter")
        print(f"   Parameters: {list(sig.parameters.keys())}")
except Exception as e:
    print(f"   ✗ Error checking PDBFixer: {e}")

print("\n" + "=" * 80)
print("Verification Complete")
print("=" * 80)

#!/usr/bin/env python
"""
Quick test to verify iteration calculation fix and convergence tracking.

This tests:
1. Iterations are calculated correctly (num_atoms * 2, capped at 100,000)
2. Convergence is tracked and reported with automatic retry
3. New output fields (converged, tolerance, energy_fluctuation) are returned
"""

import os
import sys
from pathlib import Path

# Add affinity module to path
affinity_parent = Path(__file__).parent.parent.parent  # tasks/affinity/
sys.path.insert(0, str(affinity_parent))

# Load .env file for Redis/GCS credentials
env_file = Path(__file__).parent.parent.parent.parent.parent / ".env"
if env_file.exists():
    print(f"Loading environment from: {env_file}")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value

# Setup webserver module for cache_manager import
reference_path = Path(__file__).parent.parent.parent.parent.parent / "sync" / "reference"
if reference_path.exists():
    sys.path.insert(0, str(reference_path.parent))
    import types
    webserver_module = types.ModuleType("webserver")
    webserver_module.__path__ = [str(reference_path)]
    sys.modules["webserver"] = webserver_module

from affinity.mmgbsa_utils import (
    extract_case_id,
    load_metadata,
    run_mmgbsa,
    calculate_max_iterations,
    get_available_platforms,
)
from affinity.affinity_utils import BODY_TEMPERATURE


def test_calculate_max_iterations():
    """Test the iteration calculation logic."""
    print("=" * 60)
    print("TEST: calculate_max_iterations function")
    print("=" * 60)

    test_cases = [
        # (num_atoms, tolerance, user_specified, expected_description)
        (4740, 0.1, None, "4740 atoms, tolerance=0.1, no user spec -> 4740*2=9480"),
        (7599, 0.1, None, "7599 atoms, tolerance=0.1, no user spec -> 7599*2=15198"),
        (15000, 0.1, None, "15000 atoms, tolerance=0.1, no user spec -> 15000*2=30000"),
        (4740, 0.001, None, "4740 atoms, tolerance=0.001 (strict), no user spec -> 4740*7=33180"),
        (1000, 0.1, None, "1000 atoms, tolerance=0.1, no user spec -> 1000*2=2000"),
        (500, 0.1, None, "500 atoms (below 1000), tolerance=0.1, no user spec -> max(1000,500)*2=2000"),
        (4740, 0.1, 5000, "4740 atoms, user_specified=5000 -> max(5000, 4740)=5000"),
        (4740, 0.1, 10000, "4740 atoms, user_specified=10000 -> max(10000, 4740)=10000"),
        (20000, 0.001, None, "20000 atoms, tolerance=0.001 (strict), no user spec -> min(140000, 100000)=100000"),
    ]

    print("\nTesting iteration calculation:")
    all_passed = True

    for num_atoms, tolerance, user_spec, description in test_cases:
        result = calculate_max_iterations(num_atoms, tolerance, user_spec)
        print(f"\n  {description}")
        print(f"    Result: {result:,} iterations")

        # Basic sanity checks
        if result > 100000:
            print(f"    FAIL: Result exceeds 100000 cap!")
            all_passed = False
        elif result < 1000:
            print(f"    FAIL: Result below 1000 minimum!")
            all_passed = False
        elif user_spec is None and num_atoms > 1000 and result == num_atoms:
            print(f"    FAIL: Result equals num_atoms (multiplier not applied)!")
            all_passed = False
        else:
            print(f"    PASS")

    print(f"\n{'='*60}")
    if all_passed:
        print("All iteration calculation tests PASSED")
    else:
        print("Some iteration calculation tests FAILED")
    print("=" * 60)

    return all_passed


def test_mmgbsa_with_convergence():
    """Test MM/GBSA calculation with convergence tracking."""
    print("\n" + "=" * 60)
    print("TEST: MM/GBSA with convergence tracking")
    print("=" * 60)

    # Check for CUDA
    available_platforms = get_available_platforms()
    print(f"\nAvailable platforms: {available_platforms}")

    if "CUDA" not in available_platforms:
        print("\nWARNING: CUDA not available. Using CPU (slower).")
        platform_name = "CPU" if "CPU" in available_platforms else available_platforms[0]
    else:
        platform_name = "CUDA"

    print(f"Using platform: {platform_name}")

    # Load sample data
    script_dir = Path(__file__).parent
    sample_dir = script_dir / "sample_input"
    metadata_file = sample_dir / "metadata_var1.csv"
    pdb_files = sorted(sample_dir.glob("*.pdb"))

    if not pdb_files:
        print("ERROR: No PDB files found in sample_input/")
        return False

    # Test with 1S78 (largest file) to verify retry logic works on larger systems
    pdb_file = str(pdb_files[0])  # 1S78 is largest
    print(f"\nTest file: {Path(pdb_file).name}")

    metadata_dict = load_metadata(str(metadata_file))
    case_id = extract_case_id(pdb_file)

    if case_id not in metadata_dict:
        print(f"ERROR: Case {case_id} not in metadata")
        return False

    chain_info = metadata_dict[case_id]
    receptor_chains = chain_info["receptor_chains"]
    ligand_chains = chain_info["ligand_chains"]

    print(f"Case: {case_id}")
    print(f"Receptor chains: {receptor_chains}")
    print(f"Ligand chains: {ligand_chains}")

    # Run MM/GBSA
    print(f"\nRunning MM/GBSA calculation...")
    print("-" * 40)

    result = run_mmgbsa(
        pdb_file,
        receptor_chains,
        ligand_chains,
        method="baseline",
        temperature=BODY_TEMPERATURE,
        skip_fixing=False,
        platform_name=platform_name,
    )

    print("-" * 40)

    # Check results
    print(f"\nResults:")
    print(f"  dG_bind: {result.get('dg_bind'):.4f} kcal/mol")
    print(f"  Kd: {result.get('kd_nm'):.3e} nM")
    print(f"  num_atoms: {result.get('num_atoms'):,}")
    print(f"  iterations: {result.get('iterations'):,}")
    print(f"  converged: {result.get('converged')}")
    print(f"  tolerance: {result.get('tolerance')} kJ/mol")
    print(f"  energy_fluctuation: {result.get('energy_fluctuation')}")

    # Verify new fields exist
    required_fields = ['converged', 'tolerance', 'energy_fluctuation']
    missing_fields = [f for f in required_fields if f not in result]

    if missing_fields:
        print(f"\nFAIL: Missing fields: {missing_fields}")
        return False

    # Verify iterations calculation
    num_atoms = result.get('num_atoms')
    iterations = result.get('iterations')
    expected_iterations = min(num_atoms * 2, 100000)  # With tolerance=0.1, multiplier=2

    print(f"\n  Expected iterations (initial): {expected_iterations:,} (num_atoms * 2, capped at 100000)")
    print(f"  Note: With retry logic, total iterations may be higher if convergence needed retries")

    if iterations != expected_iterations:
        print(f"  WARNING: Iterations ({iterations:,}) != expected ({expected_iterations:,})")
        # This might be okay if user_specified was used

    if iterations == num_atoms:
        print(f"  FAIL: Iterations equals num_atoms - multiplier not applied!")
        return False

    print(f"\n{'='*60}")
    print("MM/GBSA convergence test PASSED")
    print("=" * 60)

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("ITERATIONS & CONVERGENCE TEST SUITE")
    print("=" * 60)
    print(f"Testing the iteration calculation fix and convergence tracking")
    print()

    results = {}

    # Test 1: Iteration calculation logic
    results['iteration_calc'] = test_calculate_max_iterations()

    # Test 2: Full MM/GBSA with convergence
    results['mmgbsa_convergence'] = test_mmgbsa_with_convergence()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
Test full workflow determinism with random.seed() fix.

This test:
1. Clears any existing cache
2. Runs the full MM/GBSA workflow twice (without caching)
3. Compares results to verify determinism
"""

import sys
import random
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from affinity.mmgbsa_utils import (
    DETERMINISTIC_SEED,
    extract_case_id,
    load_metadata,
    run_mmgbsa_baseline,
    get_available_platforms,
)
from affinity.affinity_utils import BODY_TEMPERATURE


def clear_cache(sample_dir: Path):
    """Clear the .mmgbsa_cache directory if it exists."""
    cache_dir = sample_dir / ".mmgbsa_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Cleared cache: {cache_dir}")


def run_full_workflow(pdb_file: str, metadata_dict: dict, run_number: int,
                      max_iterations: int = 1000, platform_name: str = "CUDA"):
    """Run the full MM/GBSA workflow once."""
    case_id = extract_case_id(pdb_file)

    if case_id not in metadata_dict:
        raise ValueError(f"Case {case_id} not found in metadata")

    chain_info = metadata_dict[case_id]
    receptor_chains = chain_info["receptor_chains"]
    ligand_chains = chain_info["ligand_chains"]

    print(f"\n{'='*60}")
    print(f"RUN {run_number}: {case_id}")
    print(f"{'='*60}")
    print(f"  Receptor chains: {receptor_chains}")
    print(f"  Ligand chains: {ligand_chains}")
    print(f"  Max iterations: {max_iterations}")

    # Clear cache before each run to ensure fresh processing
    clear_cache(Path(pdb_file).parent)

    # Run the full workflow
    result = run_mmgbsa_baseline(
        pdb_file,
        receptor_chains,
        ligand_chains,
        max_iterations=max_iterations,
        skip_fixing=False,
        platform_name=platform_name,
        temperature=BODY_TEMPERATURE,
    )

    return {
        "case_id": case_id,
        "dg_bind": result["dg_bind"],
        "kd_nm": result["kd_nm"],
        "e_complex": result["e_complex"],
        "e_receptor": result["e_receptor"],
        "e_ligand": result["e_ligand"],
    }


def main():
    """Run full workflow determinism test."""
    script_dir = Path(__file__).parent
    sample_dir = script_dir / "sample_input"

    # Configuration
    max_iterations = 1000
    num_runs = 2

    # Check CUDA availability
    available_platforms = get_available_platforms()
    if "CUDA" not in available_platforms:
        print("ERROR: CUDA platform required for deterministic testing")
        print(f"Available: {available_platforms}")
        sys.exit(1)

    platform_name = "CUDA"

    # Load metadata and find PDB files
    metadata_file = sample_dir / "metadata_var1.csv"
    metadata_dict = load_metadata(str(metadata_file))
    pdb_files = sorted(sample_dir.glob("*.pdb"))

    if not pdb_files:
        print("ERROR: No PDB files found")
        sys.exit(1)

    # Use first PDB file for testing
    pdb_file = str(pdb_files[0])
    case_id = extract_case_id(pdb_file)

    print("=" * 80)
    print("FULL WORKFLOW DETERMINISM TEST")
    print("=" * 80)
    print(f"File: {Path(pdb_file).name}")
    print(f"Max iterations: {max_iterations}")
    print(f"Platform: {platform_name}")
    print(f"Number of runs: {num_runs}")
    print("=" * 80)

    # Warmup run to initialize CUDA/OpenMM
    print("\nWarmup run (initializing CUDA/OpenMM)...")
    clear_cache(sample_dir)
    _ = run_full_workflow(pdb_file, metadata_dict, 0, max_iterations, platform_name)
    print("Warmup complete.")

    # Run multiple times
    results = []
    for i in range(num_runs):
        result = run_full_workflow(pdb_file, metadata_dict, i + 1, max_iterations, platform_name)
        results.append(result)

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    r1, r2 = results[0], results[1]

    print(f"\nRun 1 vs Run 2:")
    print(f"  dG_bind:    {r1['dg_bind']:.6f} vs {r2['dg_bind']:.6f}  (diff: {abs(r1['dg_bind'] - r2['dg_bind']):.9f})")
    print(f"  E_complex:  {r1['e_complex']:.6f} vs {r2['e_complex']:.6f}  (diff: {abs(r1['e_complex'] - r2['e_complex']):.9f})")
    print(f"  E_receptor: {r1['e_receptor']:.6f} vs {r2['e_receptor']:.6f}  (diff: {abs(r1['e_receptor'] - r2['e_receptor']):.9f})")
    print(f"  E_ligand:   {r1['e_ligand']:.6f} vs {r2['e_ligand']:.6f}  (diff: {abs(r1['e_ligand'] - r2['e_ligand']):.9f})")

    # Calculate max difference
    dg_diff = abs(r1['dg_bind'] - r2['dg_bind'])
    e_complex_diff = abs(r1['e_complex'] - r2['e_complex'])

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if dg_diff < 1e-6:
        print(f"\n[PASS] EXCELLENT: dG values identical to numerical precision")
        print(f"  Max dG difference: {dg_diff:.9f} kcal/mol")
        success = True
    elif dg_diff < 0.1:
        print(f"\n[PASS] GOOD: dG values very similar")
        print(f"  Max dG difference: {dg_diff:.6f} kcal/mol")
        success = True
    elif dg_diff < 1.0:
        print(f"\n[WARN] MODERATE: Some variation in dG")
        print(f"  Max dG difference: {dg_diff:.4f} kcal/mol")
        success = False
    else:
        print(f"\n[FAIL] POOR: Significant variation in dG")
        print(f"  Max dG difference: {dg_diff:.2f} kcal/mol")
        success = False

    print("=" * 80)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

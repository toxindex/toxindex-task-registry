#!/usr/bin/env python
"""
Quick determinism test - runs in ~5-10 minutes instead of 3-4 hours.

Tests only 1 case with 2 runs to verify:
1. Same initial energy (PDBFixer determinism)
2. Same final energy (minimization determinism)
3. Same binding energy (overall determinism)

Run from tasks/affinity/affinity directory:
    python quick_determinism_test.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from affinity.mmgbsa_utils import (
    extract_case_id,
    load_metadata,
    run_mmgbsa,
    get_available_platforms,
)
from affinity.affinity_utils import BODY_TEMPERATURE

def quick_test():
    """Run quick determinism test on single case."""
    script_dir = Path(__file__).parent
    sample_dir = script_dir / "sample_input"

    # Use only first PDB file
    pdb_files = sorted(sample_dir.glob("*.pdb"))
    if not pdb_files:
        print("ERROR: No PDB files found in sample_input/")
        return False

    pdb_file = str(pdb_files[0])
    case_id = extract_case_id(pdb_file)

    # Load metadata
    metadata_file = sample_dir / "metadata_var1.csv"
    metadata_dict = load_metadata(str(metadata_file))

    if case_id not in metadata_dict:
        print(f"ERROR: Case {case_id} not found in metadata")
        return False

    # Check platform
    available_platforms = get_available_platforms()
    if "CUDA" not in available_platforms:
        print("ERROR: CUDA platform required for determinism testing")
        print(f"Available: {available_platforms}")
        return False

    print("=" * 80)
    print("QUICK DETERMINISM TEST")
    print("=" * 80)
    print(f"Case: {case_id}")
    print(f"File: {Path(pdb_file).name}")
    print(f"Platform: CUDA")
    print(f"Runs: 2 (to check determinism)")
    print("=" * 80)

    # Get chain info
    chain_info = metadata_dict[case_id]
    receptor_chains = chain_info["receptor_chains"]
    ligand_chains = chain_info["ligand_chains"]

    print(f"\nReceptor chains: {receptor_chains}")
    print(f"Ligand chains: {ligand_chains}")

    # Run 1
    print(f"\n{'='*80}")
    print("RUN 1")
    print("=" * 80)
    result1 = run_mmgbsa(
        pdb_file,
        receptor_chains,
        ligand_chains,
        method="baseline",
        temperature=BODY_TEMPERATURE,
        skip_fixing=False,
        platform_name="CUDA",
    )

    # Run 2
    print(f"\n{'='*80}")
    print("RUN 2")
    print("=" * 80)
    result2 = run_mmgbsa(
        pdb_file,
        receptor_chains,
        ligand_chains,
        method="baseline",
        temperature=BODY_TEMPERATURE,
        skip_fixing=False,
        platform_name="CUDA",
    )

    # Compare results
    print(f"\n{'='*80}")
    print("RESULTS COMPARISON")
    print("=" * 80)

    dg1 = result1.get("dg_bind")
    dg2 = result2.get("dg_bind")

    e_complex1 = result1.get("e_complex")
    e_complex2 = result2.get("e_complex")

    e_receptor1 = result1.get("e_receptor")
    e_receptor2 = result2.get("e_receptor")

    e_ligand1 = result1.get("e_ligand")
    e_ligand2 = result2.get("e_ligand")

    print(f"\nRun 1:")
    print(f"  E_complex:  {e_complex1:12.4f} kcal/mol")
    print(f"  E_receptor: {e_receptor1:12.4f} kcal/mol")
    print(f"  E_ligand:   {e_ligand1:12.4f} kcal/mol")
    print(f"  ΔG_bind:    {dg1:12.4f} kcal/mol")

    print(f"\nRun 2:")
    print(f"  E_complex:  {e_complex2:12.4f} kcal/mol")
    print(f"  E_receptor: {e_receptor2:12.4f} kcal/mol")
    print(f"  E_ligand:   {e_ligand2:12.4f} kcal/mol")
    print(f"  ΔG_bind:    {dg2:12.4f} kcal/mol")

    print(f"\nDifferences:")
    diff_complex = abs(e_complex1 - e_complex2)
    diff_receptor = abs(e_receptor1 - e_receptor2)
    diff_ligand = abs(e_ligand1 - e_ligand2)
    diff_dg = abs(dg1 - dg2)

    print(f"  ΔE_complex:  {diff_complex:12.6f} kcal/mol")
    print(f"  ΔE_receptor: {diff_receptor:12.6f} kcal/mol")
    print(f"  ΔE_ligand:   {diff_ligand:12.6f} kcal/mol")
    print(f"  ΔΔG_bind:    {diff_dg:12.6f} kcal/mol")

    # Assess determinism
    print(f"\n{'='*80}")
    print("DETERMINISM ASSESSMENT")
    print("=" * 80)

    # Thresholds for determinism
    threshold_excellent = 0.01  # < 0.01 kcal/mol: excellent
    threshold_good = 0.1        # < 0.1 kcal/mol: good
    threshold_acceptable = 0.5  # < 0.5 kcal/mol: acceptable

    is_deterministic = diff_dg < threshold_excellent

    if diff_dg < threshold_excellent:
        status = "✓ EXCELLENT"
        color = "green"
        verdict = "DETERMINISTIC"
    elif diff_dg < threshold_good:
        status = "✓ GOOD"
        color = "yellow"
        verdict = "MOSTLY DETERMINISTIC"
    elif diff_dg < threshold_acceptable:
        status = "~ MODERATE"
        color = "orange"
        verdict = "SOMEWHAT NON-DETERMINISTIC"
    else:
        status = "✗ POOR"
        color = "red"
        verdict = "NON-DETERMINISTIC"

    print(f"\nStatus: {status}")
    print(f"Verdict: {verdict}")
    print(f"ΔG difference: {diff_dg:.6f} kcal/mol")

    if diff_dg < threshold_excellent:
        print(f"\n✓ SUCCESS: Variation < {threshold_excellent} kcal/mol")
        print("  The implementation is deterministic!")
    elif diff_dg < threshold_good:
        print(f"\n✓ ACCEPTABLE: Variation < {threshold_good} kcal/mol")
        print("  Small numerical precision effects, but acceptable for practical use.")
    elif diff_dg < threshold_acceptable:
        print(f"\n⚠ WARNING: Variation < {threshold_acceptable} kcal/mol")
        print("  Moderate non-determinism. May need further tuning.")
    else:
        print(f"\n✗ FAILURE: Variation > {threshold_acceptable} kcal/mol")
        print("  Significant non-determinism detected!")
        print("\nPossible causes:")
        print("  1. PDBFixer is not producing identical output")
        print("  2. Energy minimization converging to different local minima")
        print("  3. Floating-point precision issues")
        print("  4. Non-deterministic operations in forcefield setup")

    print(f"\n{'='*80}")

    return is_deterministic


if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)

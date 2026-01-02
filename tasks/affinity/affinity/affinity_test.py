"""
Test script for affinity calculation reproducibility.

This script tests whether MM/GBSA binding affinity calculations produce
consistent results across multiple runs.

Run from the tasks/affinity/affinity directory:
    python affinity_test.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

from affinity.mmgbsa_utils import (
    extract_case_id,
    load_metadata,
    run_mmgbsa,
)
from affinity.affinity_utils import BODY_TEMPERATURE


def run_single_calculation(pdb_file: str, metadata_dict: dict, method: str, temperature: float, platform_name: str = "CUDA"):
    """
    Run a single MM/GBSA calculation.

    Returns:
        Dict with case_id, dG, Kd_nM
    """
    case_id = extract_case_id(str(pdb_file))

    if case_id not in metadata_dict:
        raise ValueError(f"Case {case_id} not found in metadata")

    chain_info = metadata_dict[case_id]
    receptor_chains = chain_info["receptor_chains"]
    ligand_chains = chain_info["ligand_chains"]

    result = run_mmgbsa(
        str(pdb_file),
        receptor_chains,
        ligand_chains,
        method=method,
        temperature=temperature,
        skip_fixing=False,
        platform_name=platform_name,
    )

    return {
        "Case": case_id,
        "dG": result.get("dg_bind"),
        "Kd_nM": result.get("kd_nm"),
    }


def test_reproducibility(pdb_file: str, metadata_dict: dict, method: str, temperature: float, num_runs: int = 3, platform_name: str = "CUDA"):
    """
    Test reproducibility by running MM/GBSA multiple times on the same input.

    Args:
        pdb_file: Path to PDB file
        metadata_dict: Metadata dictionary with chain info
        method: Method to use
        temperature: Temperature in Kelvin
        num_runs: Number of times to run the calculation
        platform_name: OpenMM platform to use (CUDA, OpenCL, CPU)

    Returns:
        Dict with reproducibility statistics
    """
    case_id = extract_case_id(str(pdb_file))
    print(f"\nTesting reproducibility for case: {case_id}")
    print(f"  Method: {method}")
    print(f"  Platform: {platform_name}")
    print(f"  Number of runs: {num_runs}")

    dg_values = []
    kd_values = []

    for i in range(num_runs):
        print(f"  Run {i + 1}/{num_runs}...", end=" ", flush=True)
        try:
            result = run_single_calculation(pdb_file, metadata_dict, method, temperature, platform_name)
            dg_values.append(result["dG"])
            kd_values.append(result["Kd_nM"])
            print(f"ΔG = {result['dG']:.4f} kcal/mol")
        except Exception as e:
            print(f"ERROR: {e}")

    if len(dg_values) < 2:
        print("  Not enough successful runs for reproducibility analysis")
        return None

    # Calculate statistics
    dg_array = np.array(dg_values)
    kd_array = np.array(kd_values)

    dg_mean = np.mean(dg_array)
    dg_std = np.std(dg_array)
    dg_range = np.max(dg_array) - np.min(dg_array)
    all_identical = np.all(dg_array == dg_array[0])

    print(f"\n  Results:")
    print(f"    All ΔG values: {dg_values}")
    print(f"    Mean ΔG: {dg_mean:.6f} kcal/mol")
    print(f"    Std Dev: {dg_std:.6f} kcal/mol")
    print(f"    Range: {dg_range:.6f} kcal/mol")
    print(f"    Identical: {all_identical}")

    return {
        "Case": case_id,
        "Method": method,
        "Num_Runs": num_runs,
        "dG_Values": dg_values,
        "dG_Mean": dg_mean,
        "dG_Std": dg_std,
        "dG_Range": dg_range,
        "Is_Identical": all_identical,
    }


def main():
    """Run reproducibility test on sample input data."""
    script_dir = Path(__file__).parent
    sample_dir = script_dir / "sample_input"

    # Configuration
    temperature = BODY_TEMPERATURE
    method = "baseline"  # Options: 'baseline', 'ensemble', 'variable_dielectric'
    num_runs = 3  # Number of times to run each calculation

    # Input files
    metadata_local = sample_dir / "metadata_var1.csv"
    pdb_files = list(sample_dir.glob("*.pdb"))

    # For faster testing, use only first PDB file (comment out to test all)
    pdb_files = pdb_files[:1]

    print("=" * 60)
    print("MM/GBSA Reproducibility Test")
    print("=" * 60)
    print(f"Method: {method}")
    print(f"Temperature: {temperature:.2f} K ({temperature - 273.15:.1f}°C)")
    print(f"Runs per file: {num_runs}")
    print(f"PDB files to test: {len(pdb_files)}")
    for pdb in pdb_files:
        print(f"  - {pdb.name}")

    # Load metadata
    print("\nLoading metadata...")
    metadata_dict = load_metadata(str(metadata_local))

    # Run reproducibility tests
    results = []
    print("\n" + "=" * 60)
    print("Running Reproducibility Tests")
    print("=" * 60)

    for pdb_file in pdb_files:
        try:
            result = test_reproducibility(
                str(pdb_file), metadata_dict, method, temperature, num_runs
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"\nError testing {pdb_file.name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Reproducibility Summary")
    print("=" * 60)

    if results:
        all_identical = all(r["Is_Identical"] for r in results)
        max_std = max(r["dG_Std"] for r in results)
        max_range = max(r["dG_Range"] for r in results)

        print(f"\nOverall Results:")
        print(f"  Cases tested: {len(results)}")
        print(f"  All results identical: {all_identical}")
        print(f"  Max standard deviation: {max_std:.6f} kcal/mol")
        print(f"  Max range: {max_range:.6f} kcal/mol")

        if all_identical:
            print("\n  ✓ MM/GBSA calculations are DETERMINISTIC")
        elif max_std < 0.001:
            print("\n  ~ MM/GBSA calculations have NEGLIGIBLE variation (<0.001 kcal/mol)")
        else:
            print("\n  ✗ MM/GBSA calculations show SIGNIFICANT variation")

        # Save detailed results
        summary_df = pd.DataFrame([
            {
                "Case": r["Case"],
                "Method": r["Method"],
                "Num_Runs": r["Num_Runs"],
                "dG_Mean": r["dG_Mean"],
                "dG_Std": r["dG_Std"],
                "dG_Range": r["dG_Range"],
                "Is_Identical": r["Is_Identical"],
                "dG_Values": str(r["dG_Values"]),
            }
            for r in results
        ])
        output_file = script_dir / "reproducibility_results.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
    else:
        print("\nNo results to display.")


if __name__ == "__main__":
    main()

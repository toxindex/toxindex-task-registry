#!/usr/bin/env python
"""
Quick test to verify CSV output includes new convergence columns.
Tests the generate_comparison_table logic without celery dependencies.
"""

import pandas as pd
from pathlib import Path


def generate_comparison_table(results_by_method: dict, groundtruth_df=None):
    """
    Copy of generate_comparison_table from affinity_celery.py for testing.
    """
    # Start with case IDs from all methods
    all_cases = set()
    for df in results_by_method.values():
        all_cases.update(df['Case'].values)

    # Don't sort by Case here - we'll sort by ranking at the end
    comparison_df = pd.DataFrame({'Case': list(all_cases)})

    # Extract metadata columns (same for all methods, use first available)
    metadata_cols = ['receptor_chains', 'ligand_chains', 'num_atoms', 'iterations', 'converged', 'tolerance', 'energy_fluctuation', 'energy_before', 'energy_after']
    for col in metadata_cols:
        for df in results_by_method.values():
            if col in df.columns:
                comparison_df = pd.merge(comparison_df, df[['Case', col]],
                                        on='Case', how='left')
                break  # Only need from one method (same for all)

    # Add method predictions (both dG and Kd)
    for method_name, df in results_by_method.items():
        method_col_dg = f'{method_name}_dG'
        method_col_kd = f'{method_name}_Kd_nM'
        # Merge dG
        comparison_df = pd.merge(comparison_df, df[['Case', 'Predicted_dG']],
                                on='Case', how='left')
        comparison_df = comparison_df.rename(columns={'Predicted_dG': method_col_dg})
        # Merge Kd if available
        if 'Predicted_Kd_nM' in df.columns:
            comparison_df = pd.merge(comparison_df, df[['Case', 'Predicted_Kd_nM']],
                                    on='Case', how='left')
            comparison_df = comparison_df.rename(columns={'Predicted_Kd_nM': method_col_kd})

    # Add groundtruth if provided
    if groundtruth_df is not None:
        comparison_df = pd.merge(comparison_df, groundtruth_df[['Case', 'Experimental_dG']],
                                on='Case', how='left')
        comparison_df = comparison_df.rename(columns={'Experimental_dG': 'groundtruth_dG'})

    # Add rankings for each method
    for method_name in results_by_method.keys():
        method_col = f'{method_name}_dG'
        rank_col = f'{method_name}_rank'
        # Rank by dG (lower/more negative = better binding = rank 1)
        comparison_df[rank_col] = comparison_df[method_col].rank(ascending=True, method='min')
        # Store ranks as integers (nullable to allow missing)
        comparison_df[rank_col] = comparison_df[rank_col].astype("Int64")

    # Round dG columns; format Kd columns in scientific notation to avoid 0.0000 for tiny values
    dg_cols = []
    kd_cols = []
    for col in comparison_df.columns:
        if col.endswith('_dG') or col == 'groundtruth_dG':
            dg_cols.append(col)
        elif col.endswith('_Kd_nM'):
            kd_cols.append(col)
    if dg_cols:
        comparison_df[dg_cols] = comparison_df[dg_cols].round(3)
    if kd_cols:
        # format as strings in scientific notation with 3 significant figures
        comparison_df[kd_cols] = comparison_df[kd_cols].map(
            lambda x: f"{x:.3e}" if pd.notnull(x) else x
        )

    # Reorder columns: Case, metadata, groundtruth (if exists), methods (dG, Kd), rankings
    cols = ['Case']
    # Add metadata columns
    for col in metadata_cols:
        if col in comparison_df.columns:
            cols.append(col)
    if 'groundtruth_dG' in comparison_df.columns:
        cols.append('groundtruth_dG')
    # Add dG columns
    for m in results_by_method.keys():
        cols.append(f'{m}_dG')
        if f'{m}_Kd_nM' in comparison_df.columns:
            cols.append(f'{m}_Kd_nM')
    # Add rankings
    cols.extend([f'{m}_rank' for m in results_by_method.keys()])

    comparison_df = comparison_df[cols]

    return comparison_df


def test_csv_output():
    """Test that CSV output includes convergence columns."""
    print("=" * 60)
    print("TEST: CSV Output Format")
    print("=" * 60)

    # Create mock results similar to what affinity_single_pdb returns
    mock_results = [
        {
            "Case": "2DD8",
            "Predicted_dG": -81.6417,
            "Predicted_Kd_nM": 2.922e-49,
            "receptor_chains": "H,L",
            "ligand_chains": "S",
            "num_atoms": 9322,
            "iterations": 18644,
            "converged": True,
            "tolerance": 0.1,
            "energy_fluctuation": 0.003242,
            "energy_before": 7586.67,
            "energy_after": -17455.83,
        },
        {
            "Case": "1S78",
            "Predicted_dG": -65.1234,
            "Predicted_Kd_nM": 1.5e-35,
            "receptor_chains": "A",
            "ligand_chains": "B",
            "num_atoms": 7599,
            "iterations": 15198,
            "converged": False,
            "tolerance": 0.1,
            "energy_fluctuation": 0.025,  # Above threshold
            "energy_before": 91828.24,
            "energy_after": -31035.28,
        },
    ]

    # Convert to DataFrame (simulating what happens in affinity_aggregate_method)
    results_df = pd.DataFrame(mock_results)

    print("\nInput DataFrame:")
    print(results_df.to_string())
    print(f"\nColumns: {list(results_df.columns)}")

    # Test generate_comparison_table
    results_by_method = {"baseline": results_df}

    print("\n" + "-" * 40)
    print("Generating comparison table...")
    comparison_df = generate_comparison_table(results_by_method, groundtruth_df=None)

    print("\nOutput DataFrame:")
    print(comparison_df.to_string())
    print(f"\nColumns: {list(comparison_df.columns)}")

    # Check required columns exist
    required_cols = ['converged', 'tolerance', 'energy_fluctuation', 'energy_before', 'energy_after']
    missing_cols = [col for col in required_cols if col not in comparison_df.columns]

    if missing_cols:
        print(f"\nFAIL: Missing columns: {missing_cols}")
        return False

    print(f"\nPASS: All convergence columns present")

    # Save sample CSV
    output_file = Path(__file__).parent / "sample_affinity_comparison.csv"
    comparison_df.to_csv(output_file, index=False)
    print(f"\nSample CSV saved to: {output_file}")

    # Show CSV content
    print("\nCSV Content:")
    print("-" * 60)
    with open(output_file) as f:
        print(f.read())
    print("-" * 60)

    return True


if __name__ == "__main__":
    import sys
    success = test_csv_output()
    sys.exit(0 if success else 1)

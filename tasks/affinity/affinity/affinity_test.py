"""
Test script for affinity calculation reproducibility, symmetry, and independence.

This script tests:
1. Reproducibility: Same input produces same output across multiple runs
2. Symmetry: Swapping receptor/ligand chains produces identical results
3. Independence: Results don't depend on other cases in the batch

Run from the tasks/affinity/affinity directory:
    python affinity_test.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from affinity.mmgbsa_utils import (
    extract_case_id,
    load_metadata,
    run_mmgbsa,
    _normalize_chains,
    get_available_platforms,
    get_platform_name,
)
from affinity.affinity_utils import BODY_TEMPERATURE
import openmm


class ProgressReporter(openmm.MinimizationReporter):
    """Progress reporter for energy minimization."""
    def __init__(self, max_iterations: int, report_interval: int = 100):
        super().__init__()
        self.max_iterations = max_iterations
        self.report_interval = report_interval
        self.last_reported = 0
    
    def report(self, iteration, x, grad, args):
        """Report progress every N iterations.
        
        Args:
            iteration: Current iteration number
            x: Current particle positions
            grad: Current gradient
            args: SWIG-wrapped map with statistics like 'system energy', 'restraint energy', etc.
        """
        if iteration % self.report_interval == 0 or iteration == self.max_iterations - 1:
            progress = (iteration + 1) / self.max_iterations * 100
            # Try to get energy from args (SWIG map supports [] indexing)
            try:
                energy_kj = args['system energy']
                energy_kcal = energy_kj / 4.184  # Convert kJ/mol to kcal/mol
                print(f"    Progress: {iteration + 1:,}/{self.max_iterations:,} iterations ({progress:.1f}%) - Energy: {energy_kcal:.2f} kcal/mol", flush=True)
            except (KeyError, TypeError):
                # If energy not available, just show iteration count
                print(f"    Progress: {iteration + 1:,}/{self.max_iterations:,} iterations ({progress:.1f}%)", flush=True)
            self.last_reported = iteration
        return False  # Don't stop minimization


def run_single_calculation(pdb_file: str, metadata_dict: dict, method: str, 
                          temperature: float, platform_name: str = "CUDA"):
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
        "receptor_chains": receptor_chains,
        "ligand_chains": ligand_chains,
    }


def test_minimization_determinism(pdb_file: str, metadata_dict: dict, method: str,
                                 temperature: float, num_runs: int = 3,
                                 platform_name: str = "CUDA"):
    """
    Diagnostic test: Run minimization once, then calculate energies multiple times.

    This isolates whether variation comes from:
    - Minimization step (if this test passes but normal test fails)
    - Energy calculation step (if this test also fails)

    Args:
        pdb_file: Path to PDB file
        metadata_dict: Metadata dictionary with chain info
        method: Method to use
        temperature: Temperature in Kelvin
        num_runs: Number of times to recalculate energies
        platform_name: OpenMM platform to use (CUDA, OpenCL, CPU)

    Returns:
        Dict with diagnostic results
    """
    case_id = extract_case_id(str(pdb_file))
    print(f"\n[DIAGNOSTIC] Testing energy calculation determinism for case: {case_id}")
    print(f"  Approach: Minimize ONCE, then calculate energies {num_runs} times")
    print(f"  This isolates whether variation is from minimization or energy calculation")

    # Import here to avoid circular dependency
    from affinity.mmgbsa_utils import run_mmgbsa_baseline
    import openmm
    from openmm import app, unit
    import pdbfixer

    # Get chain info
    if case_id not in metadata_dict:
        raise ValueError(f"Case {case_id} not found in metadata")

    chain_info = metadata_dict[case_id]
    receptor_chains = _normalize_chains(chain_info["receptor_chains"])
    ligand_chains = _normalize_chains(chain_info["ligand_chains"])

    print(f"  Receptor chains: {receptor_chains}")
    print(f"  Ligand chains: {ligand_chains}")

    # Step 1: Minimize ONCE
    print(f"\n  Step 1: Performing minimization ONCE...")
    # We'll do minimization manually to save the state
    # (This duplicates some code from mmgbsa_utils but needed for diagnostics)

    # Fix PDB
    fixer = pdbfixer.PDBFixer(filename=str(pdb_file))
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    # Set deterministic seed for adding missing atoms
    from affinity.mmgbsa_utils import DETERMINISTIC_SEED
    fixer.addMissingAtoms(seed=DETERMINISTIC_SEED)
    fixer.addMissingHydrogens(pH=7.0)

    # Create system
    from openmm import app as openmm_app
    forcefield = openmm_app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
    modeller = openmm_app.Modeller(fixer.topology, fixer.positions)
    system = forcefield.createSystem(modeller.topology,
                                    nonbondedMethod=openmm_app.NoCutoff,
                                    constraints=openmm_app.HBonds)

    # Create integrator and simulation
    integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName(platform_name)

    # Set CUDA precision
    if platform_name == "CUDA":
        platform.setPropertyDefaultValue('CudaPrecision', 'double')
        platform.setPropertyDefaultValue('CudaDeviceIndex', '0')
        platform.setPropertyDefaultValue('DeterministicForces', 'true')

    simulation = openmm_app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    # Minimize
    num_atoms = len(list(modeller.topology.atoms()))
    from affinity.mmgbsa_utils import calculate_max_iterations
    tolerance = 0.001 * unit.kilojoule_per_mole / unit.nanometer
    # For testing, use 1000 iterations for faster runs
    # Note: This may not fully converge for large systems, but sufficient for determinism testing
    max_iterations = 1000

    print(f"  Minimizing {num_atoms} atoms (max {max_iterations} iterations)...")
    # Add progress reporter to show progress every 100 iterations
    reporter = ProgressReporter(max_iterations, report_interval=100)
    simulation.minimizeEnergy(maxIterations=max_iterations, tolerance=tolerance, reporter=reporter)

    # Get minimized positions
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    minimized_positions = state.getPositions()

    print(f"  ✓ Minimization complete. Now calculating energies {num_runs} times...")

    # Step 2: Calculate energies multiple times using SAME minimized positions
    dg_values = []

    for i in range(num_runs):
        print(f"  Calculation {i + 1}/{num_runs}...", end=" ", flush=True)

        # Calculate complex energy (from saved minimized positions)
        simulation.context.setPositions(minimized_positions)
        state = simulation.context.getState(getEnergy=True)
        e_complex = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

        # Calculate receptor energy
        rec_modeller = openmm_app.Modeller(modeller.topology, minimized_positions)
        to_delete = [c for c in rec_modeller.topology.chains() if c.id not in receptor_chains]
        rec_modeller.delete(to_delete)

        rec_system = forcefield.createSystem(rec_modeller.topology,
                                            nonbondedMethod=openmm_app.NoCutoff,
                                            constraints=openmm_app.HBonds)
        rec_integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
        rec_platform = openmm.Platform.getPlatformByName(platform_name)
        if platform_name == "CUDA":
            rec_platform.setPropertyDefaultValue('CudaPrecision', 'double')
            rec_platform.setPropertyDefaultValue('CudaDeviceIndex', '0')
            rec_platform.setPropertyDefaultValue('DeterministicForces', 'true')
        rec_sim = openmm_app.Simulation(rec_modeller.topology, rec_system, rec_integrator, rec_platform)
        rec_sim.context.setPositions(rec_modeller.positions)
        e_receptor = rec_sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

        # Calculate ligand energy
        lig_modeller = openmm_app.Modeller(modeller.topology, minimized_positions)
        to_delete = [c for c in lig_modeller.topology.chains() if c.id not in ligand_chains]
        lig_modeller.delete(to_delete)

        lig_system = forcefield.createSystem(lig_modeller.topology,
                                            nonbondedMethod=openmm_app.NoCutoff,
                                            constraints=openmm_app.HBonds)
        lig_integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
        lig_platform = openmm.Platform.getPlatformByName(platform_name)
        if platform_name == "CUDA":
            lig_platform.setPropertyDefaultValue('CudaPrecision', 'double')
            lig_platform.setPropertyDefaultValue('CudaDeviceIndex', '0')
            lig_platform.setPropertyDefaultValue('DeterministicForces', 'true')
        lig_sim = openmm_app.Simulation(lig_modeller.topology, lig_system, lig_integrator, lig_platform)
        lig_sim.context.setPositions(lig_modeller.positions)
        e_ligand = lig_sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

        # Calculate dG
        dg = e_complex - (e_receptor + e_ligand)
        dg_values.append(dg)
        print(f"ΔG = {dg:.6f} kcal/mol")

    # Analyze results
    dg_array = np.array(dg_values)
    dg_std = np.std(dg_array)
    dg_range = np.max(dg_array) - np.min(dg_array)
    all_identical = np.allclose(dg_array, dg_array[0], rtol=1e-10, atol=1e-6)

    print(f"\n  [DIAGNOSTIC RESULTS]")
    print(f"    All ΔG values: {dg_values}")
    print(f"    Std Dev: {dg_std:.9f} kcal/mol")
    print(f"    Range: {dg_range:.9f} kcal/mol")
    print(f"    All identical (< 1e-6): {all_identical}")

    if all_identical:
        print(f"    ✓ Energy calculation is DETERMINISTIC")
        print(f"    → Variation in full test must come from MINIMIZATION step")
    else:
        print(f"    ✗ Energy calculation is NON-DETERMINISTIC")
        print(f"    → Problem is in energy calculation, not just minimization")

    return {
        "Case": case_id,
        "dG_Values": dg_values,
        "dG_Std": dg_std,
        "dG_Range": dg_range,
        "Is_Deterministic": all_identical,
    }


def test_reproducibility(pdb_file: str, metadata_dict: dict, method: str,
                        temperature: float, num_runs: int = 3,
                        platform_name: str = "CUDA"):
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
            result = run_single_calculation(pdb_file, metadata_dict, method, 
                                          temperature, platform_name)
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
    # With deterministic settings (double precision, deterministic forces, random seed):
    # - Expect variations < 0.01 kcal/mol for identical runs (numerical precision only)
    # - Small variations (< 0.1 kcal/mol) still acceptable for practical purposes
    # - Variations > 0.5 kcal/mol indicate non-deterministic behavior or convergence issues
    all_identical = np.allclose(dg_array, dg_array[0], rtol=1e-5, atol=0.01)

    print(f"\n  Results:")
    print(f"    All ΔG values: {dg_values}")
    print(f"    Mean ΔG: {dg_mean:.6f} kcal/mol")
    print(f"    Std Dev: {dg_std:.6f} kcal/mol")
    print(f"    Range: {dg_range:.6f} kcal/mol")
    print(f"    Deterministic (within 0.01 kcal/mol): {all_identical}")
    if not all_identical:
        if dg_std < 0.01:
            print(f"    ✓ EXCELLENT: Variation < 0.01 kcal/mol (nearly perfect determinism)")
        elif dg_std < 0.1:
            print(f"    ✓ GOOD: Variation < 0.1 kcal/mol (acceptable numerical precision)")
        elif dg_std < 0.5:
            print(f"    ~ MODERATE: Variation < 0.5 kcal/mol (may need tuning)")
        else:
            print(f"    ✗ POOR: Variation > 0.5 kcal/mol (non-deterministic behavior)")

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


def test_symmetry(pdb_file: str, metadata_dict_original: dict, 
                 metadata_dict_swapped: dict, method: str, 
                 temperature: float, platform_name: str = "CUDA"):
    """
    Test symmetry: swapping receptor/ligand chains should produce identical results.
    
    Mathematically: ΔG = E_complex - (E_receptor + E_ligand) 
                    = E_complex - (E_ligand + E_receptor)
    So swapping should produce the same result.
    
    Args:
        pdb_file: Path to PDB file
        metadata_dict_original: Original metadata (receptor/ligand as specified)
        metadata_dict_swapped: Swapped metadata (receptor<->ligand swapped)
        method: Method to use
        temperature: Temperature in Kelvin
        platform_name: OpenMM platform to use
    
    Returns:
        Dict with symmetry test results
    """
    case_id = extract_case_id(str(pdb_file))
    print(f"\nTesting symmetry for case: {case_id}")
    print(f"  Method: {method}")
    print(f"  Platform: {platform_name}")
    
    if case_id not in metadata_dict_original or case_id not in metadata_dict_swapped:
        print(f"  ERROR: Case {case_id} not found in both metadata dicts")
        return None
    
    # Get original chains
    orig_rec = metadata_dict_original[case_id]["receptor_chains"]
    orig_lig = metadata_dict_original[case_id]["ligand_chains"]
    
    # Get swapped chains
    swap_rec = metadata_dict_swapped[case_id]["receptor_chains"]
    swap_lig = metadata_dict_swapped[case_id]["ligand_chains"]
    
    print(f"  Original: receptor={orig_rec}, ligand={orig_lig}")
    print(f"  Swapped:  receptor={swap_rec}, ligand={swap_lig}")
    
    # Verify they are actually swapped
    orig_rec_sorted = sorted(orig_rec)
    orig_lig_sorted = sorted(orig_lig)
    swap_rec_sorted = sorted(swap_rec)
    swap_lig_sorted = sorted(swap_lig)
    
    if not (set(orig_rec_sorted) == set(swap_lig_sorted) and 
            set(orig_lig_sorted) == set(swap_rec_sorted)):
        print(f"  WARNING: Chains don't appear to be swapped!")
        print(f"    Original receptor chains: {orig_rec_sorted}")
        print(f"    Swapped ligand chains: {swap_lig_sorted}")
        print(f"    Original ligand chains: {orig_lig_sorted}")
        print(f"    Swapped receptor chains: {swap_rec_sorted}")
    
    # Run both calculations
    print(f"  Running original calculation...", end=" ", flush=True)
    try:
        result_orig = run_single_calculation(pdb_file, metadata_dict_original, 
                                           method, temperature, platform_name)
        dg_orig = result_orig["dG"]
        print(f"ΔG = {dg_orig:.4f} kcal/mol")
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    
    print(f"  Running swapped calculation...", end=" ", flush=True)
    try:
        result_swap = run_single_calculation(pdb_file, metadata_dict_swapped, 
                                           method, temperature, platform_name)
        dg_swap = result_swap["dG"]
        print(f"ΔG = {dg_swap:.4f} kcal/mol")
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    
    # Compare results
    # Swapping chains SHOULD produce identical results mathematically:
    # dG = E_complex - (E_receptor + E_ligand) = E_complex - (E_ligand + E_receptor)
    # With deterministic settings, expect differences < 0.1 kcal/mol
    dg_diff = abs(dg_orig - dg_swap)
    dg_relative_diff = dg_diff / max(abs(dg_orig), abs(dg_swap), 1e-10)
    is_identical = np.isclose(dg_orig, dg_swap, rtol=1e-5, atol=0.1)
    
    print(f"\n  Symmetry Test Results:")
    print(f"    Original ΔG: {dg_orig:.6f} kcal/mol")
    print(f"    Swapped ΔG:  {dg_swap:.6f} kcal/mol")
    print(f"    Difference:  {dg_diff:.6e} kcal/mol")
    print(f"    Relative diff: {dg_relative_diff:.2e}")
    print(f"    Symmetric (within 0.1 kcal/mol): {is_identical}")

    if is_identical:
        print(f"    ✓ SYMMETRY CONFIRMED: Swapping chains produces identical results")
    else:
        if dg_diff < 0.1:
            print(f"    ✓ GOOD: Small difference ({dg_diff:.4f} kcal/mol) - acceptable")
        elif dg_diff < 0.5:
            print(f"    ~ MODERATE: Difference ({dg_diff:.4f} kcal/mol) - may need tuning")
        else:
            print(f"    ✗ SYMMETRY VIOLATION: Large difference ({dg_diff:.4f} kcal/mol)")
            print(f"      This indicates non-deterministic behavior or implementation issues")
    
    return {
        "Case": case_id,
        "Method": method,
        "Original_dG": dg_orig,
        "Swapped_dG": dg_swap,
        "Difference": dg_diff,
        "Relative_Difference": dg_relative_diff,
        "Is_Identical": is_identical,
        "Original_receptor": orig_rec,
        "Original_ligand": orig_lig,
        "Swapped_receptor": swap_rec,
        "Swapped_ligand": swap_lig,
    }


def test_independence(pdb_files: List[str], metadata_dict_full: dict, 
                     metadata_dict_partial: dict, method: str,
                     temperature: float, platform_name: str = "CUDA"):
    """
    Test independence: Results for overlapping cases should be identical
    regardless of other cases in the batch.
    
    This verifies that each case is calculated independently and there's no
    global state pollution or non-deterministic behavior affecting results.
    
    Args:
        pdb_files: List of PDB file paths
        metadata_dict_full: Metadata with all cases
        metadata_dict_partial: Metadata with subset of cases
        method: Method to use
        temperature: Temperature in Kelvin
        platform_name: OpenMM platform to use
    
    Returns:
        Dict with independence test results
    """
    print(f"\nTesting independence (batch size effect)")
    print(f"  Method: {method}")
    print(f"  Platform: {platform_name}")
    
    # Find overlapping cases
    full_cases = set(metadata_dict_full.keys())
    partial_cases = set(metadata_dict_partial.keys())
    overlapping_cases = full_cases & partial_cases
    
    if not overlapping_cases:
        print(f"  ERROR: No overlapping cases found")
        return None
    
    print(f"  Overlapping cases: {sorted(overlapping_cases)}")
    
    results = []
    
    for case_id in sorted(overlapping_cases):
        # Find corresponding PDB file
        pdb_file = None
        for pdb in pdb_files:
            if extract_case_id(str(pdb)) == case_id:
                pdb_file = pdb
                break
        
        if not pdb_file:
            print(f"  WARNING: No PDB file found for case {case_id}")
            continue
        
        print(f"\n  Testing case: {case_id}")
        
        # Run with full metadata
        print(f"    Running with full metadata...", end=" ", flush=True)
        try:
            result_full = run_single_calculation(pdb_file, metadata_dict_full, 
                                                method, temperature, platform_name)
            dg_full = result_full["dG"]
            print(f"ΔG = {dg_full:.4f} kcal/mol")
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        
        # Run with partial metadata
        print(f"    Running with partial metadata...", end=" ", flush=True)
        try:
            result_partial = run_single_calculation(pdb_file, metadata_dict_partial, 
                                                    method, temperature, platform_name)
            dg_partial = result_partial["dG"]
            print(f"ΔG = {dg_partial:.4f} kcal/mol")
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        
        # Compare
        # Results should be independent of batch size
        # With deterministic settings, expect differences < 0.1 kcal/mol
        dg_diff = abs(dg_full - dg_partial)
        dg_relative_diff = dg_diff / max(abs(dg_full), abs(dg_partial), 1e-10)
        is_identical = np.isclose(dg_full, dg_partial, rtol=1e-5, atol=0.1)
        
        print(f"    Difference: {dg_diff:.6e} kcal/mol (relative: {dg_relative_diff:.2e})")
        if is_identical:
            print(f"    ✓ INDEPENDENCE CONFIRMED")
        else:
            print(f"    ✗ INDEPENDENCE VIOLATION: Results differ!")
        
        results.append({
            "Case": case_id,
            "Full_dG": dg_full,
            "Partial_dG": dg_partial,
            "Difference": dg_diff,
            "Relative_Difference": dg_relative_diff,
            "Is_Identical": is_identical,
        })
    
    if not results:
        return None
    
    all_identical = all(r["Is_Identical"] for r in results)
    max_diff = max(r["Difference"] for r in results)
    
    print(f"\n  Independence Test Summary:")
    print(f"    Cases tested: {len(results)}")
    print(f"    All identical: {all_identical}")
    print(f"    Max difference: {max_diff:.6e} kcal/mol")
    
    return {
        "Method": method,
        "Cases_Tested": len(results),
        "All_Identical": all_identical,
        "Max_Difference": max_diff,
        "Results": results,
    }


def main():
    """Run comprehensive tests on sample input data."""
    script_dir = Path(__file__).parent
    sample_dir = script_dir / "sample_input"

    # Configuration
    temperature = BODY_TEMPERATURE
    method = "baseline"  # Options: 'baseline', 'ensemble', 'variable_dielectric'
    num_runs = 3  # Number of times to run each calculation for reproducibility
    
    # Input files
    metadata_var1 = sample_dir / "metadata_var1.csv"
    metadata_var2 = sample_dir / "metadata_var2.csv"
    metadata_var3 = sample_dir / "metadata_var3.csv"
    pdb_files = sorted(sample_dir.glob("*.pdb"))

    # Check available platforms and REQUIRE CUDA for deterministic testing
    available_platforms = get_available_platforms()
    print("=" * 80)
    print("MM/GBSA Comprehensive Test Suite")
    print("=" * 80)
    print(f"Available OpenMM platforms: {', '.join(available_platforms) if available_platforms else 'None'}")

    # REQUIRE CUDA for deterministic testing
    if "CUDA" not in available_platforms:
        print("\n" + "=" * 80)
        print("ERROR: CUDA platform is REQUIRED for deterministic testing!")
        print("=" * 80)
        print(f"Available platforms: {', '.join(available_platforms)}")
        print("\nReasons:")
        print("  - CUDA supports 'DeterministicForces' flag for reproducible results")
        print("  - OpenCL and CPU platforms may give non-deterministic results")
        print("  - Without CUDA, tests will fail due to numerical variations")
        print("\nTo fix:")
        print("  1. Install CUDA-enabled OpenMM: conda install -c conda-forge openmm cudatoolkit")
        print("  2. Or use a machine with NVIDIA GPU and CUDA support")
        print("  3. Verify CUDA: python -c 'import openmm; print(openmm.Platform.getPlatformByName(\"CUDA\"))'")
        print("\n" + "=" * 80)
        raise RuntimeError("CUDA platform not available. Deterministic testing requires CUDA.")

    platform_name = "CUDA"
    print(f"✓ Using platform: CUDA (GPU acceleration with deterministic forces)")
    
    print(f"Method: {method}")
    print(f"Temperature: {temperature:.2f} K ({temperature - 273.15:.1f}°C)")
    print(f"Platform: {platform_name}")
    print(f"Runs per file (reproducibility): {num_runs}")
    print(f"\nPDB files:")
    for pdb in pdb_files:
        print(f"  - {pdb.name}")
    print(f"\nMetadata files:")
    print(f"  - {metadata_var1.name}")
    print(f"  - {metadata_var2.name} (swapped chains)")
    print(f"  - {metadata_var3.name} (subset)")

    # Load metadata
    print("\n" + "=" * 80)
    print("Loading Metadata")
    print("=" * 80)
    metadata_dict_var1 = load_metadata(str(metadata_var1))
    metadata_dict_var2 = load_metadata(str(metadata_var2))
    metadata_dict_var3 = load_metadata(str(metadata_var3))
    
    print(f"\nMetadata var1 cases: {sorted(metadata_dict_var1.keys())}")
    print(f"Metadata var2 cases: {sorted(metadata_dict_var2.keys())}")
    print(f"Metadata var3 cases: {sorted(metadata_dict_var3.keys())}")

    all_results = {
        "diagnostic": [],
        "reproducibility": [],
        "symmetry": [],
        "independence": None,
    }

    # Test 0: DIAGNOSTIC - Isolate source of variation
    print("\n" + "=" * 80)
    print("DIAGNOSTIC TEST: Energy Calculation Determinism")
    print("=" * 80)
    print("This test minimizes ONCE, then calculates energies multiple times.")
    print("Purpose: Determine if variation comes from minimization or energy calculation.")
    print("")

    for pdb_file in pdb_files[:1]:  # Just test first file for diagnostic
        case_id = extract_case_id(str(pdb_file))
        if case_id in metadata_dict_var1:
            try:
                result = test_minimization_determinism(
                    str(pdb_file), metadata_dict_var1, method, temperature,
                    num_runs, platform_name
                )
                if result:
                    all_results["diagnostic"].append(result)
            except Exception as e:
                print(f"\nError in diagnostic test for {pdb_file.name}: {e}")
                import traceback
                traceback.print_exc()

    # Test 1: Reproducibility
    print("\n" + "=" * 80)
    print("TEST 1: Reproducibility")
    print("=" * 80)
    print("Testing if same input produces same output across multiple runs...")
    
    for pdb_file in pdb_files:
        case_id = extract_case_id(str(pdb_file))
        if case_id in metadata_dict_var1:
            try:
                result = test_reproducibility(
                    str(pdb_file), metadata_dict_var1, method, temperature, 
                    num_runs, platform_name
                )
                if result:
                    all_results["reproducibility"].append(result)
            except Exception as e:
                print(f"\nError testing reproducibility for {pdb_file.name}: {e}")

    # Test 2: Symmetry
    print("\n" + "=" * 80)
    print("TEST 2: Symmetry")
    print("=" * 80)
    print("Testing if swapping receptor/ligand chains produces identical results...")
    print("(Mathematically: ΔG = E_complex - (E_receptor + E_ligand)")
    print("                = E_complex - (E_ligand + E_receptor) ✓)")
    
    for pdb_file in pdb_files:
        case_id = extract_case_id(str(pdb_file))
        if case_id in metadata_dict_var1 and case_id in metadata_dict_var2:
            try:
                result = test_symmetry(
                    str(pdb_file), metadata_dict_var1, metadata_dict_var2,
                    method, temperature, platform_name
                )
                if result:
                    all_results["symmetry"].append(result)
            except Exception as e:
                print(f"\nError testing symmetry for {pdb_file.name}: {e}")

    # Test 3: Independence
    print("\n" + "=" * 80)
    print("TEST 3: Independence")
    print("=" * 80)
    print("Testing if results are independent of other cases in batch...")
    print("(Reducing from 3 cases to 2 should produce same results for overlapping cases)")
    
    try:
        independence_result = test_independence(
            [str(pdb) for pdb in pdb_files],
            metadata_dict_var1,  # Full (3 cases)
            metadata_dict_var3,   # Partial (2 cases)
            method, temperature, platform_name
        )
        all_results["independence"] = independence_result
    except Exception as e:
        print(f"\nError testing independence: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    # Diagnostic summary
    if all_results["diagnostic"]:
        diag_results = all_results["diagnostic"]
        all_diag_deterministic = all(r["Is_Deterministic"] for r in diag_results)
        max_diag_std = max(r["dG_Std"] for r in diag_results)

        print(f"\n0. DIAGNOSTIC (Energy Calculation Determinism):")
        print(f"   Cases tested: {len(diag_results)}")
        print(f"   All deterministic: {all_diag_deterministic}")
        print(f"   Max std dev: {max_diag_std:.9f} kcal/mol")

        if all_diag_deterministic:
            print(f"   ✓ Energy calculation is DETERMINISTIC")
            print(f"   → Any variation in tests below comes from MINIMIZATION step")
        else:
            print(f"   ✗ Energy calculation has variation")
            print(f"   → Problem is not just minimization, also energy calculation")

    # Reproducibility summary
    if all_results["reproducibility"]:
        repro_results = all_results["reproducibility"]
        all_repro_identical = all(r["Is_Identical"] for r in repro_results)
        max_repro_std = max(r["dG_Std"] for r in repro_results)
        max_repro_range = max(r["dG_Range"] for r in repro_results)

        print(f"\n1. Reproducibility:")
        print(f"   Cases tested: {len(repro_results)}")
        print(f"   All results deterministic: {all_repro_identical}")
        print(f"   Max standard deviation: {max_repro_std:.6f} kcal/mol")
        print(f"   Max range: {max_repro_range:.6f} kcal/mol")

        if all_repro_identical:
            print(f"   ✓ EXCELLENT: Deterministic (< 0.01 kcal/mol variation)")
        elif max_repro_std < 0.1:
            print(f"   ✓ GOOD: Acceptable variation (< 0.1 kcal/mol)")
        elif max_repro_std < 0.5:
            print(f"   ~ MODERATE: Some variation (< 0.5 kcal/mol)")
        else:
            print(f"   ✗ POOR: Significant variation (> 0.5 kcal/mol)")

    # Symmetry summary
    if all_results["symmetry"]:
        sym_results = all_results["symmetry"]
        all_sym_identical = all(r["Is_Identical"] for r in sym_results)
        max_sym_diff = max(r["Difference"] for r in sym_results)

        print(f"\n2. Symmetry:")
        print(f"   Cases tested: {len(sym_results)}")
        print(f"   All results identical: {all_sym_identical}")
        print(f"   Max difference: {max_sym_diff:.6e} kcal/mol")
        
        if all_sym_identical:
            print(f"   ✓ SYMMETRIC: Swapping chains produces identical results")
        else:
            print(f"   ✗ ASYMMETRIC: Results differ when chains are swapped")
            print(f"     This may indicate implementation issues or numerical precision problems")

    # Independence summary
    if all_results["independence"]:
        indep_result = all_results["independence"]
        print(f"\n3. Independence:")
        print(f"   Cases tested: {indep_result['Cases_Tested']}")
        print(f"   All results identical: {indep_result['All_Identical']}")
        print(f"   Max difference: {indep_result['Max_Difference']:.6e} kcal/mol")
        
        if indep_result['All_Identical']:
            print(f"   ✓ INDEPENDENT: Results don't depend on other cases in batch")
        else:
            print(f"   ✗ DEPENDENT: Results vary based on other cases")
            print(f"     This indicates global state pollution or non-deterministic behavior")

    # Save detailed results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    output_dir = script_dir
    if all_results["reproducibility"]:
        repro_df = pd.DataFrame([
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
            for r in all_results["reproducibility"]
        ])
        repro_file = output_dir / "test_reproducibility_results.csv"
        repro_df.to_csv(repro_file, index=False)
        print(f"Reproducibility results saved to: {repro_file}")
    
    if all_results["symmetry"]:
        sym_df = pd.DataFrame(all_results["symmetry"])
        sym_file = output_dir / "test_symmetry_results.csv"
        sym_df.to_csv(sym_file, index=False)
        print(f"Symmetry results saved to: {sym_file}")
    
    if all_results["independence"]:
        indep_df = pd.DataFrame(all_results["independence"]["Results"])
        indep_file = output_dir / "test_independence_results.csv"
        indep_df.to_csv(indep_file, index=False)
        print(f"Independence results saved to: {indep_file}")

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

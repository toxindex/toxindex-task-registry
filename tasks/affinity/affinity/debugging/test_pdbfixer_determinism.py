#!/usr/bin/env python
"""
Test if PDBFixer produces deterministic output with our changes.

This test runs in ~30 seconds and checks if the root cause is fixed.
"""

import sys
import random
from pathlib import Path
import numpy as np

# Add parent directory to path (tasks/affinity/ so we can import affinity.*)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from affinity.mmgbsa_utils import DETERMINISTIC_SEED, normalize_coordinates

# Set random seed BEFORE importing OpenMM/PDBFixer to catch any initialization randomness
random.seed(DETERMINISTIC_SEED)

import pdbfixer
from openmm import app, unit


def process_pdb_once(pdb_path):
    """Process PDB file once and return positions."""
    # Reset random state at start of each processing run for full determinism
    random.seed(DETERMINISTIC_SEED)

    # Step 1: PDBFixer (without addMissingHydrogens)
    fixer = pdbfixer.PDBFixer(filename=pdb_path)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=DETERMINISTIC_SEED)
    # DO NOT call addMissingHydrogens() - that's non-deterministic!

    topology = fixer.topology
    positions = fixer.positions

    # Step 2: Normalize coordinates
    positions = normalize_coordinates(positions, precision=6)

    # Step 3: Create modeller and add hydrogens using forcefield
    forcefield = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
    modeller = app.Modeller(topology, positions)

    # CRITICAL: Use double-precision CUDA platform for deterministic hydrogen placement
    # Per OpenMM docs: "double-precision CUDA will result in deterministic simulations"
    # Single-precision CUDA and CPU platforms are NOT deterministic.
    from openmm import Platform
    try:
        hydrogen_platform = Platform.getPlatformByName("CUDA")
        hydrogen_platform.setPropertyDefaultValue('CudaPrecision', 'double')
        print("(CUDA double)", end=" ", flush=True)
    except Exception:
        hydrogen_platform = Platform.getPlatformByName("Reference")
        print("(Reference)", end=" ", flush=True)

    # Set Python's random seed before addHydrogens for deterministic initial placement
    random.seed(DETERMINISTIC_SEED)
    modeller.addHydrogens(forcefield, pH=7.0, platform=hydrogen_platform)

    # Step 4: Normalize again after adding hydrogens
    modeller.positions = normalize_coordinates(modeller.positions, precision=6)

    # Step 5: Create system and get initial energy
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds
    )

    from openmm import VerletIntegrator, Platform
    integrator = VerletIntegrator(2.0 * unit.femtoseconds)

    try:
        platform = Platform.getPlatformByName("CUDA")
        platform.setPropertyDefaultValue('CudaPrecision', 'double')
        platform.setPropertyDefaultValue('CudaDeviceIndex', '0')
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        simulation = app.Simulation(modeller.topology, system, integrator, platform)
    except:
        platform = Platform.getPlatformByName("CPU")
        simulation = app.Simulation(modeller.topology, system, integrator, platform)

    simulation.context.setPositions(modeller.positions)
    state = simulation.context.getState(getEnergy=True, getPositions=True)

    energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    positions_array = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    return energy, positions_array, len(list(modeller.topology.atoms()))


def test_pdbfixer_determinism():
    """Test if PDBFixer produces identical output on multiple runs."""
    script_dir = Path(__file__).parent
    sample_dir = script_dir / "sample_input"

    pdb_files = sorted(sample_dir.glob("*.pdb"))
    if not pdb_files:
        print("ERROR: No PDB files found")
        return False

    pdb_file = str(pdb_files[0])

    print("=" * 80)
    print("PDBFIXER DETERMINISM TEST")
    print("=" * 80)
    print(f"File: {Path(pdb_file).name}")
    print(f"Testing if PDBFixer + hydrogen addition is deterministic...")
    print("=" * 80)

    # Warmup run: The first OpenMM simulation triggers lazy initialization
    # (CUDA context, forcefield parsing, etc.) that consumes random numbers.
    # After warmup, random.seed() in process_pdb_once gives perfect determinism.
    print("\nWarmup run (discarded)...", end=" ", flush=True)
    _ = process_pdb_once(pdb_file)
    print("done")

    # Run 2 times
    results = []
    for i in range(2):
        print(f"\nRun {i+1}/2...", end=" ", flush=True)
        energy, positions, num_atoms = process_pdb_once(pdb_file)
        print(f"E_initial = {energy:.2f} kcal/mol ({num_atoms} atoms)")
        results.append((energy, positions, num_atoms))

    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON")
    print("=" * 80)

    energy1, pos1, natoms1 = results[0]
    energy2, pos2, natoms2 = results[1]

    print(f"\nInitial Energies:")
    print(f"  Run 1: {energy1:.6f} kcal/mol")
    print(f"  Run 2: {energy2:.6f} kcal/mol")

    energy_diff = abs(energy1 - energy2)

    print(f"\nEnergy Difference:")
    print(f"  |E1 - E2|: {energy_diff:.9f} kcal/mol")

    max_energy_diff = energy_diff

    # Check positions
    pos_diff = np.max(np.abs(pos1 - pos2))

    print(f"\nPosition Difference (max coordinate diff):")
    print(f"  |Pos1 - Pos2|: {pos_diff:.9e} nm")

    max_pos_diff = pos_diff

    # Verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print("=" * 80)

    # Thresholds
    energy_threshold = 1.0  # If initial energies differ by > 1 kcal/mol, PDBFixer is non-deterministic
    pos_threshold = 1e-6    # If positions differ by > 1e-6 nm, there's variation

    if max_energy_diff < 1e-6:
        print(f"\n✓ EXCELLENT: Initial energies identical to numerical precision")
        print(f"  Max difference: {max_energy_diff:.9f} kcal/mol")
        verdict = "DETERMINISTIC"
        success = True
    elif max_energy_diff < 0.1:
        print(f"\n✓ GOOD: Initial energies very similar")
        print(f"  Max difference: {max_energy_diff:.6f} kcal/mol")
        print(f"  (Small variation acceptable from floating-point precision)")
        verdict = "MOSTLY DETERMINISTIC"
        success = True
    elif max_energy_diff < energy_threshold:
        print(f"\n~ MODERATE: Initial energies show some variation")
        print(f"  Max difference: {max_energy_diff:.4f} kcal/mol")
        verdict = "SOME VARIATION"
        success = False
    else:
        print(f"\n✗ POOR: Initial energies differ significantly")
        print(f"  Max difference: {max_energy_diff:.2f} kcal/mol")
        print(f"  PDBFixer is producing different structures on each run!")
        verdict = "NON-DETERMINISTIC"
        success = False

    if max_pos_diff > pos_threshold:
        print(f"\n⚠ WARNING: Atomic positions differ by {max_pos_diff:.3e} nm")
        if max_pos_diff > 1e-3:
            print(f"  This is significant variation (> 0.001 nm)")
            success = False

    print(f"\nOverall: {verdict}")
    print("=" * 80)

    return success


if __name__ == "__main__":
    success = test_pdbfixer_determinism()
    sys.exit(0 if success else 1)

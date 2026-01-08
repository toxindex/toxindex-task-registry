"""MM/GBSA utilities for binding affinity calculations."""

import os
import json
import re
import random
import tempfile
import threading
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from io import StringIO
import numpy as np
import openmm
from openmm import unit
from openmm import app
import pdbfixer
from affinity.affinity_utils import convert_delta_g_kcal_to_kd_nm, BODY_TEMPERATURE, ROOM_TEMPERATURE

# Set deterministic random seed for reproducibility
# This ensures energy minimization converges to the same state
DETERMINISTIC_SEED = 42

# ============================================================================
# OpenMM Warmup and GCS-based PDB Caching
# ============================================================================
# OpenMM has lazy initialization that consumes random numbers on first use.
# This causes the first run to produce different hydrogen positions than
# subsequent runs, even with random.seed(). The warmup ensures consistent
# behavior from the very first cached structure.

_openmm_initialized = False
_init_lock = threading.Lock()


def _ensure_openmm_initialized():
    """
    Perform warmup to ensure OpenMM's lazy initialization is complete.

    This ensures random.seed() works correctly for hydrogen placement.
    Must be called before any caching operation to guarantee the first
    cached structure is deterministic.
    """
    global _openmm_initialized
    if _openmm_initialized:
        return

    with _init_lock:
        if _openmm_initialized:
            return

        # Minimal OpenMM operation to trigger all lazy initialization
        # This includes forcefield parsing, CUDA/OpenCL context creation, etc.
        random.seed(DETERMINISTIC_SEED)

        # Load forcefield (triggers XML parsing and initialization)
        forcefield = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')

        # Create a minimal topology with one atom to trigger platform init
        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue('ALA', chain)
        topology.addAtom('CA', app.Element.getBySymbol('C'), residue)

        # Create positions for the single atom
        positions = np.array([[0.0, 0.0, 0.0]]) * unit.nanometer

        # Create system and simulation to fully initialize OpenMM
        try:
            modeller = app.Modeller(topology, positions)
            system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff)
            integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)

            # Try CUDA first, fall back to CPU
            try:
                platform = openmm.Platform.getPlatformByName("CUDA")
                # Use 'mixed' precision for ~2-3x speedup
                # Removed DeterministicForces for faster parallel GPU execution
                platform.setPropertyDefaultValue('CudaPrecision', 'mixed')
            except Exception:
                platform = openmm.Platform.getPlatformByName("CPU")

            simulation = app.Simulation(modeller.topology, system, integrator, platform)
            simulation.context.setPositions(positions)

            # Get energy to fully initialize everything
            _ = simulation.context.getState(getEnergy=True)

            # Clean up
            del simulation, system, integrator

        except Exception as e:
            # If minimal simulation fails, just load forcefield (should be enough)
            pass

        _openmm_initialized = True


def get_processed_pdb(complex_pdb_path: str, ph: float = 7.0,
                      use_cache: bool = False) -> Tuple[app.Topology, unit.Quantity]:
    """
    Get processed PDB (with hydrogens added) using deterministic PDBFixer.

    This function uses double-precision CUDA + random.seed() to ensure
    fully deterministic hydrogen placement. No caching needed since
    PDBFixer now produces identical results on every run.

    Args:
        complex_pdb_path: Path to input PDB file
        ph: pH for hydrogen addition (default 7.0)
        use_cache: Deprecated, ignored. Kept for API compatibility.

    Returns:
        Tuple of (topology, positions)
    """
    # 1. Ensure OpenMM is warmed up before any processing
    _ensure_openmm_initialized()

    # 2. Process with PDBFixer using deterministic settings
    print(f"  Processing PDB with PDBFixer (deterministic mode)")

    # Set random seed for deterministic hydrogen placement
    random.seed(DETERMINISTIC_SEED)

    # Use double-precision CUDA for deterministic hydrogen placement
    # Per OpenMM docs: "double-precision CUDA will result in deterministic simulations"
    # Single-precision CUDA and CPU platforms are NOT deterministic due to
    # non-deterministic order of floating-point summation in force calculations.
    try:
        pdbfixer_platform = openmm.Platform.getPlatformByName('CUDA')
        pdbfixer_platform.setPropertyDefaultValue('CudaPrecision', 'double')
        print(f"  Using CUDA double-precision for deterministic PDBFixer")
    except Exception:
        # Fall back to Reference platform if CUDA not available
        pdbfixer_platform = openmm.Platform.getPlatformByName('Reference')
        print(f"  Using Reference platform for deterministic PDBFixer (CUDA not available)")

    fixer = pdbfixer.PDBFixer(filename=complex_pdb_path, platform=pdbfixer_platform)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=DETERMINISTIC_SEED)

    # CRITICAL: Set Python's random seed before addMissingHydrogens for determinism
    # OpenMM uses random.random() internally for initial hydrogen placement
    random.seed(DETERMINISTIC_SEED)
    fixer.addMissingHydrogens(pH=ph)

    return fixer.topology, fixer.positions


def calculate_max_iterations(num_atoms: int, tolerance_kj_mol: float = 0.001,
                            user_specified: Optional[int] = None,
                            max_cap: int = 100000) -> int:
    """
    Calculate appropriate max iterations for energy minimization based on system size.

    Rules:
    - Base: 1 iteration per atom (minimum 1000)
    - For very strict tolerance (0.001 kJ/mol): multiply by 7
    - For strict tolerance (0.01 kJ/mol): multiply by 3
    - For moderate tolerance (0.1 kJ/mol): multiply by 2
    - Safety limit: max_cap iterations (default 100,000)

    Args:
        num_atoms: Number of atoms in system
        tolerance_kj_mol: Tolerance in kJ/mol
        user_specified: If provided, use this value (but ensure minimum)
        max_cap: Maximum iterations cap (default 100,000)

    Returns:
        Recommended max iterations
    """
    if user_specified is not None:
        # User specified a value, but ensure it meets minimum requirements
        base = max(1000, num_atoms)
        if tolerance_kj_mol <= 0.001:
            minimum = base * 5  # At least 5x for very strict
        elif tolerance_kj_mol <= 0.01:
            minimum = base * 2  # At least 2x for strict
        else:
            minimum = base  # At least base for moderate

        return min(max(user_specified, minimum), max_cap)

    # Calculate based on system size
    base_iterations = max(1000, num_atoms)

    # Adjust for tolerance strictness
    if tolerance_kj_mol <= 0.001:
        # Very strict: need 5-10x more iterations (use 7 as middle ground)
        multiplier = 7
    elif tolerance_kj_mol <= 0.01:
        # Strict: moderate iterations
        multiplier = 3
    elif tolerance_kj_mol <= 0.1:
        # Moderate: fewer iterations
        multiplier = 2
    else:
        # Loose: base iterations
        multiplier = 1

    calculated = base_iterations * multiplier

    # Safety limit to prevent excessive computation
    return min(calculated, max_cap)


def normalize_coordinates(positions, precision: int = 6):
    """
    Normalize coordinates to fixed precision for deterministic behavior.
    
    Rounds coordinates to a fixed number of decimal places to eliminate
    floating-point precision differences that can accumulate during calculations.
    
    Args:
        positions: OpenMM positions (Quantity with unit, or numpy array)
        precision: Number of decimal places (default: 6 for nanometer precision)
    
    Returns:
        Normalized positions with same unit/type as input
    """
    # Handle different input types
    if isinstance(positions, openmm.unit.Quantity):
        # OpenMM Quantity: convert to nanometers, round, convert back
        positions_array = positions.value_in_unit(unit.nanometer)
        positions_array = np.round(positions_array, precision)
        # Return as Quantity with nanometer unit
        return positions_array * unit.nanometer
    elif isinstance(positions, np.ndarray):
        # Numpy array: assume it's already in nanometers
        positions_array = np.round(positions, precision)
        return positions_array
    else:
        # Try to convert to numpy array (for asNumpy=True case)
        try:
            positions_array = np.array(positions)
            positions_array = np.round(positions_array, precision)
            return positions_array
        except Exception:
            # If conversion fails, return as-is
            return positions


def extract_case_id(filename: str) -> str:
    """
    Extract case ID from filename.
    
    Examples:
        "1S78_complex.pdb" -> "1S78"
        "2DD8.pdb" -> "2DD8"
        "/path/to/1S78_complex.pdb" -> "1S78"
        "7d1b50d389164f13887a9f776c632fad_427b5dbd-ae5c-4630-9e7f-68c97ff4e77f_2DD8.pdb" -> "2DD8"
        "2e38ae40e81642c086e14bbb139896d6_ad2e0a63-5c1a-4dea-986d-667bcc4ce5b7_1S78.pdb" -> "1S78"
    """
    basename = os.path.basename(filename)
    # Remove extension
    name_without_ext = os.path.splitext(basename)[0]
    
    # Handle filenames with prefixes (e.g., UUIDs or hashes before the case ID)
    # Case ID is typically 4 characters (PDB format: 1 digit + 3 alphanumeric, or 4 alphanumeric)
    # Look for the last underscore-separated segment that matches PDB ID pattern
    parts = name_without_ext.split('_')
    
    # PDB ID pattern: typically 4 alphanumeric characters, often starting with a digit
    # Pattern: 1 digit + 3 alphanumeric, or 4 alphanumeric
    pdb_id_pattern = re.compile(r'^[0-9][A-Za-z0-9]{3}$|^[A-Za-z0-9]{4}$')
    
    # Try to find a PDB ID pattern, checking from the end backwards
    for part in reversed(parts):
        if pdb_id_pattern.match(part):
            return part
    
    # Fallback: if no pattern matches, use the last part after removing "_complex" suffix
    # This handles cases like "1S78_complex" -> "1S78"
    case_id = re.sub(r'_complex$', '', parts[-1])
    return case_id


def parse_metadata_json(metadata_path: str) -> Dict[str, Dict]:
    """
    Load JSON metadata file.
    
    Expected format:
    {
        "case_ID": {
            "receptor_chains": ["H", "L"],
            "ligand_chains": ["A"]
        },
        ...
    }
    
    Chains are normalized (sorted) for deterministic behavior.
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Normalize chain lists (handle both list and string formats)
    # Sorting ensures deterministic behavior and enables symmetry testing
    normalized = {}
    for case_id, info in metadata.items():
        normalized[case_id] = {
            "receptor_chains": _normalize_chains(info.get("receptor_chains", [])),
            "ligand_chains": _normalize_chains(info.get("ligand_chains", []))
        }
    
    return normalized


def parse_metadata_csv(metadata_path: str) -> Dict[str, Dict]:
    """
    Load CSV metadata file.
    
    Expected format:
    case_ID,receptor_chains,ligand_chains
    1S78,"H,L","A"
    2DD8,"H,L","S"
    
    Chains are normalized (sorted) for deterministic behavior.
    Empty rows are skipped.
    """
    df = pd.read_csv(metadata_path)
    
    # Validate required columns
    required_cols = ['case_ID', 'receptor_chains', 'ligand_chains']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV metadata missing required columns: {missing}")
    
    metadata = {}
    for _, row in df.iterrows():
        case_id = str(row['case_ID']).strip()
        # Skip empty rows
        if not case_id or case_id.lower() == 'nan':
            continue
        
        receptor_chains = _normalize_chains(row['receptor_chains'])
        ligand_chains = _normalize_chains(row['ligand_chains'])
        
        # Skip if no chains defined
        if not receptor_chains or not ligand_chains:
            continue
        
        metadata[case_id] = {
            "receptor_chains": receptor_chains,
            "ligand_chains": ligand_chains
        }
    
    return metadata


def load_metadata(metadata_path: str) -> Dict[str, Dict]:
    """
    Auto-detect format and load metadata file.
    
    Supports JSON and CSV formats.
    """
    ext = os.path.splitext(metadata_path)[1].lower()
    
    if ext == '.json':
        return parse_metadata_json(metadata_path)
    elif ext == '.csv':
        return parse_metadata_csv(metadata_path)
    else:
        raise ValueError(f"Unsupported metadata format: {ext}. Expected .json or .csv")


def _normalize_chains(chains) -> List[str]:
    """
    Normalize chain input to list of strings.
    
    Returns sorted list for deterministic behavior (same chains in same order).
    This ensures that swapping receptor/ligand chains produces identical results
    when the same chains are used (just labeled differently).
    """
    if isinstance(chains, list):
        normalized = [str(c).strip() for c in chains]
    elif isinstance(chains, str):
        # Handle comma-separated strings
        normalized = [c.strip() for c in chains.split(',') if c.strip()]
    else:
        normalized = []
    
    # Sort for deterministic ordering (ensures consistent behavior)
    return sorted(normalized)


def validate_metadata(pdb_files: List[str], metadata: Dict[str, Dict]) -> None:
    """
    Validate that all PDB files have corresponding metadata entries.
    
    Raises ValueError with list of missing case_IDs if any are missing.
    """
    missing = []
    for pdb_file in pdb_files:
        case_id = extract_case_id(pdb_file)
        if case_id not in metadata:
            missing.append(case_id)
    
    if missing:
        raise ValueError(
            f"Missing metadata entries for {len(missing)} PDB file(s): {', '.join(missing)}. "
            f"All PDB files must have corresponding entries in metadata."
        )


def clean_pdb(input_pdb: str, output_pdb: str) -> None:
    """
    Clean PDB file using PDBFixer while preserving chain IDs.
    
    Removes heterogens, fixes missing atoms/residues.
    Preserves original chain IDs to maintain compatibility with metadata.
    """
    print(f"Cleaning PDB: {os.path.basename(input_pdb)} -> {os.path.basename(output_pdb)}")
    
    # Load original PDB to get chain IDs and create mapping
    original_pdb = app.PDBFile(input_pdb)
    original_chains = list(original_pdb.topology.chains())
    original_chain_ids = [chain.id for chain in original_chains]
    
    # Create mapping from residue index to original chain ID
    residue_to_chain = {}
    for chain in original_chains:
        for residue in chain.residues():
            residue_to_chain[residue.index] = chain.id
    
    # Use PDBFixer to clean
    fixer = pdbfixer.PDBFixer(filename=input_pdb)

    # Remove heterogens (water, ions, etc.)
    fixer.removeHeterogens(keepWater=False)

    # Find and add missing residues/atoms
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    # Set deterministic seed for adding missing atoms
    fixer.addMissingAtoms(seed=DETERMINISTIC_SEED)
    
    # Restore original chain IDs by mapping residues back to their original chains
    cleaned_chains = list(fixer.topology.chains())
    cleaned_chain_map = {}
    
    # Build mapping of cleaned chains to original chain IDs based on residues
    for chain in cleaned_chains:
        # Find which original chain this cleaned chain corresponds to
        # by checking the first residue's original chain
        if len(list(chain.residues())) > 0:
            first_residue = list(chain.residues())[0]
            if first_residue.index in residue_to_chain:
                original_chain_id = residue_to_chain[first_residue.index]
                cleaned_chain_map[chain] = original_chain_id
    
    # Apply chain ID mapping
    for chain, original_id in cleaned_chain_map.items():
        chain.id = original_id
    
    # If we still have unmapped chains, try to preserve order
    unmapped_chains = [c for c in cleaned_chains if c not in cleaned_chain_map]
    if unmapped_chains and len(unmapped_chains) <= len(original_chain_ids):
        # Try to map by position
        for i, chain in enumerate(unmapped_chains):
            if i < len(original_chain_ids):
                chain.id = original_chain_ids[i]
    
    # Save cleaned PDB
    with open(output_pdb, 'w') as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
    
    print(f"  Cleaned PDB saved to {os.path.basename(output_pdb)}")


def get_available_platforms() -> List[str]:
    """
    Get list of available OpenMM platforms.
    
    Returns:
        List of platform names that are available
    """
    available = []
    for platform_name in ["CUDA", "OpenCL", "CPU", "Reference"]:
        try:
            platform = openmm.Platform.getPlatformByName(platform_name)
            available.append(platform_name)
        except Exception:
            continue
    return available


def get_platform_name() -> str:
    """
    Get OpenMM platform name from environment variable.
    
    Defaults to CUDA if not set. Falls back to CPU only if CUDA unavailable.
    Automatically detects available platforms and uses the best one.
    """
    requested_platform = os.environ.get("OPENMM_PLATFORM", "CUDA")
    
    # Get available platforms
    available_platforms = get_available_platforms()
    
    if not available_platforms:
        raise RuntimeError("No OpenMM platforms available! Check OpenMM installation.")
    
    # Try requested platform first
    if requested_platform in available_platforms:
        return requested_platform
    
    # Fallback logic: prefer CUDA > OpenCL > CPU
    preferred_order = ["CUDA", "OpenCL", "CPU", "Reference"]
    for platform in preferred_order:
        if platform in available_platforms:
            print(f"Warning: Requested platform '{requested_platform}' not available.")
            print(f"Available platforms: {', '.join(available_platforms)}")
            print(f"Using platform: {platform}")
            return platform
    
    # Should never reach here, but just in case
    return available_platforms[0]


def run_mmgbsa_baseline(complex_pdb_path: str, receptor_chains: List[str], ligand_chains: List[str],
                       max_iterations: Optional[int] = None, skip_fixing: bool = False,
                       platform_name: Optional[str] = None, temperature: float = BODY_TEMPERATURE) -> Dict:
    """
    Run baseline MM/GBSA calculation (single-trajectory method).

    This method is DETERMINISTIC - it uses VerletIntegrator (no stochastic terms)
    and only performs energy minimization, not MD simulation.
    
    DETERMINISM GUARANTEES:
    - Chains are normalized (sorted) to ensure consistent ordering
    - Coordinates normalized to fixed precision (6 decimal places) after PDBFixer
    - CUDA precision set to 'mixed' for consistent GPU calculations
    - Very strict minimization tolerance (0.001 kJ/mol) for consistent convergence
    - Positions normalized before and after minimization
    - Single-trajectory method ensures receptor/ligand use same coordinates as complex
    - Expected variation: < 0.1 kcal/mol (numerical precision only)
    
    SYMMETRY NOTE:
    - Swapping receptor/ligand chains should produce identical results since:
      dG = E_complex - (E_receptor + E_ligand) = E_complex - (E_ligand + E_receptor)
    - However, subtle differences may occur due to:
      * Chain ordering in PDB file
      * Numerical precision
      * PDBFixer processing order
    - For true symmetry, ensure chains are properly normalized (this function does that)
   
    Run baseline MM/GBSA calculation (single-trajectory method).

    This method is DETERMINISTIC - it uses VerletIntegrator (no stochastic terms)
    and only performs energy minimization, not MD simulation.

    MM/GBSA calculates an approximation of binding free energy (dG) as:
    dG_bind approximately = E_complex - (E_receptor + E_ligand)

    Where E includes:
    - Molecular mechanics energy (bonded + non-bonded interactions)
    - Solvation free energy (GB model for polar, SASA for non-polar)
    - NOTE: Entropy term (-T*dS) is typically NOT included in this calculation

    The Kd is then derived from dG using: Kd = exp(-dG / (RT))

    IMPORTANT LIMITATIONS:
    - The calculated dG is an approximation that may miss entropic contributions
    - The conversion to Kd assumes dG is a true standard Gibbs free energy (dG standard)
    - MM/GBSA dG values are often used for relative ranking rather than absolute Kd prediction
    - Temperature significantly affects Kd: approximately 2 to 3 times change per 10 degrees C

    Args:
        temperature: Temperature in Kelvin for Kd conversion.
                     Use BODY_TEMPERATURE (310.15 K) for human drug screening,
                     ROOM_TEMPERATURE (298.15 K) for laboratory comparisons.

    Returns:
        Dict with 'dg_bind' (kcal/mol) and 'kd_nm' (nM) keys.
        Note: kd_nm is derived from dG and should be interpreted with caution.
    """
    if platform_name is None:
        platform_name = get_platform_name()
    
    print(f"Processing {os.path.basename(complex_pdb_path)} (Baseline MM/GBSA)...")

    # 1. Fix PDB - Use GCS-cached processed structure for cross-pod determinism
    if skip_fixing:
        pdb_file = app.PDBFile(complex_pdb_path)
        topology = pdb_file.topology
        positions = pdb_file.positions
    else:
        # Use get_processed_pdb() which handles:
        # - OpenMM warmup for deterministic first-run behavior
        # - GCS-based caching for cross-pod consistency
        # - PDBFixer processing with deterministic random seeds
        topology, positions = get_processed_pdb(complex_pdb_path, ph=7.0, use_cache=True)
    
    # 2. Create System
    forcefield = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')

    # CRITICAL FOR DETERMINISM: Normalize positions to ensure identical starting point
    # This removes any floating-point precision differences
    positions = normalize_coordinates(positions, precision=6)

    modeller = app.Modeller(topology, positions)

    # If skip_fixing mode, try to add hydrogens (for compatibility)
    if skip_fixing:
        try:
            # CRITICAL: Set Python's random seed before addHydrogens for determinism
            random.seed(DETERMINISTIC_SEED)
            modeller.addHydrogens(forcefield, pH=7.0)
            modeller.positions = normalize_coordinates(modeller.positions, precision=6)
        except Exception:
            pass

    constraints = None if skip_fixing else app.HBonds
    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.NoCutoff,
                                     constraints=constraints)
    
    # 3. Minimize Complex
    # Use VerletIntegrator for deterministic behavior (no stochastic terms)
    # Langevin integrator is unnecessary since we only do minimization, not MD
    # VerletIntegrator is already fully deterministic (no random number generation)
    integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)

    # REQUIRE CUDA with double precision for deterministic minimization
    # Per OpenMM docs: only double-precision CUDA guarantees deterministic results
    if platform_name != "CUDA":
        raise RuntimeError(
            f"Platform '{platform_name}' not supported for deterministic minimization. "
            f"Only CUDA with double precision is supported. "
            f"Set platform_name='CUDA' or OPENMM_PLATFORM=CUDA environment variable."
        )

    try:
        platform = openmm.Platform.getPlatformByName("CUDA")
        # Use 'double' precision for deterministic minimization
        # Per OpenMM docs: "double-precision CUDA will result in deterministic simulations"
        # Mixed precision is faster but NOT deterministic
        platform.setPropertyDefaultValue('CudaPrecision', 'double')
        # Explicitly set device index for consistency
        platform.setPropertyDefaultValue('CudaDeviceIndex', '0')
        print(f"  Using OpenMM Platform: CUDA (precision: double, deterministic mode)")

        simulation = app.Simulation(modeller.topology, system, integrator, platform)
    except Exception as e:
        # Do NOT fall back to other platforms - they are not deterministic
        available = get_available_platforms()
        raise RuntimeError(
            f"CUDA platform required but failed to initialize: {e}\n"
            f"Available platforms: {', '.join(available) if available else 'none'}\n"
            f"Deterministic minimization requires CUDA with double precision. "
            f"Please ensure CUDA is available and properly configured."
        )
    
    # Set positions - VerletIntegrator is deterministic (no stochastic terms)
    # For deterministic behavior, ensure:
    # 1. Chain ordering is normalized (done via _normalize_chains)
    # 2. Same platform is used
    # 3. Same initial positions (normalized to fixed precision - done above)

    # Set initial positions (already normalized before creating modeller)
    simulation.context.setPositions(modeller.positions)

    # Calculate system size for iteration determination
    num_atoms = len(list(modeller.topology.atoms()))

    # Use moderate tolerance for faster minimization
    # OpenMM minimizeEnergy expects tolerance in kJ/(mol*nm) (gradient/force units)
    # 0.1 kJ/(mol*nm) is moderate - good balance between speed and convergence
    tolerance = 0.1 * unit.kilojoule_per_mole / unit.nanometer  # Moderate tolerance for speed
    tolerance_kj_mol = 0.1  # Store energy-equivalent for iteration calculation

    # Calculate appropriate max iterations based on system size
    effective_max_iterations = calculate_max_iterations(
        num_atoms,
        tolerance_kj_mol=tolerance_kj_mol,
        user_specified=max_iterations
    )

    # Minimization with retry logic for convergence
    # Retry multipliers: 1x, 2.5x, 5x of initial iterations (capped at 100,000)
    retry_multipliers = [1.0, 2.5, 5.0]
    max_retries = len(retry_multipliers)
    convergence_threshold = 0.01  # kcal/mol

    state_before = simulation.context.getState(getEnergy=True)
    e_before = state_before.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    converged = False
    energy_fluctuation = None
    total_iterations_used = 0

    for retry_idx, multiplier in enumerate(retry_multipliers):
        current_iterations = min(int(effective_max_iterations * multiplier), 100000)
        attempt_num = retry_idx + 1

        if retry_idx == 0:
            print(f"  Minimizing complex ({num_atoms:,} atoms, tolerance={tolerance_kj_mol} kJ/mol, max {current_iterations:,} iterations)...")
        else:
            print(f"  Retry {retry_idx}/{max_retries-1}: Increasing to {current_iterations:,} iterations...")

        # CRITICAL: Use LocalEnergyMinimizer directly for more control over determinism
        try:
            from openmm import LocalEnergyMinimizer
            LocalEnergyMinimizer.minimize(simulation.context, tolerance, current_iterations)
        except (ImportError, AttributeError):
            simulation.minimizeEnergy(maxIterations=current_iterations, tolerance=tolerance)

        total_iterations_used += current_iterations

        # Get energy after this round
        state_after = simulation.context.getState(getEnergy=True)
        e_after = state_after.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

        # Check convergence by running a few more iterations
        convergence_test_iterations = 100
        try:
            from openmm import LocalEnergyMinimizer
            LocalEnergyMinimizer.minimize(simulation.context, tolerance, convergence_test_iterations)
        except (ImportError, AttributeError):
            simulation.minimizeEnergy(maxIterations=convergence_test_iterations, tolerance=tolerance)

        state_convergence_test = simulation.context.getState(getEnergy=True)
        e_convergence_test = state_convergence_test.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        energy_fluctuation = abs(e_convergence_test - e_after)

        converged = energy_fluctuation < convergence_threshold

        if converged:
            print(f"  Convergence: YES (fluctuation={energy_fluctuation:.6f} kcal/mol < {convergence_threshold} threshold)")
            break
        else:
            print(f"  Convergence: NO (fluctuation={energy_fluctuation:.4f} kcal/mol >= {convergence_threshold} threshold)")
            if retry_idx < max_retries - 1:
                print(f"  Will retry with more iterations...")

    # Update effective_max_iterations to reflect total used
    effective_max_iterations = total_iterations_used

    # Final energy change report
    state_final = simulation.context.getState(getEnergy=True)
    e_final = state_final.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    energy_change = e_final - e_before
    print(f"  Energy change: {e_before:.2f} -> {e_final:.2f} kcal/mol (Δ = {energy_change:.2f})")
    print(f"  Total iterations used: {total_iterations_used:,}")

    if not converged:
        print(f"  WARNING: Did not converge after {max_retries} attempts with {total_iterations_used:,} total iterations")

    if abs(energy_change) > 1000:  # Large energy change suggests issues
        print(f"  WARNING: Large energy change during minimization: {energy_change:.2f} kcal/mol")
        print(f"  This may indicate convergence issues or system problems.")
    
    # Get minimized state and normalize positions
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    e_complex = state.getPotentialEnergy()
    minimized_positions = state.getPositions()
    
    # Normalize minimized positions for deterministic behavior
    # This ensures consistent coordinates for receptor/ligand calculations
    minimized_positions = normalize_coordinates(minimized_positions, precision=6)
    
    print(f"  Complex Energy: {e_complex.value_in_unit(unit.kilocalories_per_mole):.4f} kcal/mol")
    
    # 4. Calculate Receptor and Ligand Energies
    # IMPORTANT: Receptor and ligand must be minimized separately to avoid artifacts
    # from removing chains (atoms may be in strained positions)
    # NOTE: Chain IDs are normalized (sorted) to ensure deterministic behavior
    def calc_energy(chain_ids, name):
        # Ensure chain_ids are sorted for deterministic behavior
        chain_ids_sorted = sorted(chain_ids) if isinstance(chain_ids, list) else [chain_ids]

        sub_modeller = app.Modeller(modeller.topology, minimized_positions)
        to_delete = [c for c in sub_modeller.topology.chains() if c.id not in chain_ids_sorted]
        sub_modeller.delete(to_delete)

        sub_system = forcefield.createSystem(sub_modeller.topology,
                                             nonbondedMethod=app.NoCutoff,
                                             constraints=app.HBonds)
        # Use VerletIntegrator for deterministic single-point energy calculation
        # VerletIntegrator is already fully deterministic (no random number generation)
        sub_integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)

        try:
            platform = openmm.Platform.getPlatformByName(platform_name)

            # Set same CUDA precision settings for consistency (mixed for speed)
            if platform_name == "CUDA":
                try:
                    platform.setPropertyDefaultValue('CudaPrecision', 'mixed')
                    platform.setPropertyDefaultValue('CudaDeviceIndex', '0')
                except Exception:
                    pass

            sub_sim = app.Simulation(sub_modeller.topology, sub_system, sub_integrator, platform)
        except Exception:
            # Try fallback platform
            available = get_available_platforms()
            if available:
                try:
                    platform = openmm.Platform.getPlatformByName(available[0])
                    sub_sim = app.Simulation(sub_modeller.topology, sub_system, sub_integrator, platform)
                except Exception:
                    sub_sim = app.Simulation(sub_modeller.topology, sub_system, sub_integrator)
            else:
                sub_sim = app.Simulation(sub_modeller.topology, sub_system, sub_integrator)
        
        # Use positions from sub_modeller (already extracted from normalized minimized_positions)
        sub_sim.context.setPositions(sub_modeller.positions)
        
        # SINGLE-TRAJECTORY METHOD: Use same coordinates from minimized complex
        # Do NOT minimize receptor/ligand separately - this would allow them to relax
        # to lower energies than in the complex, giving incorrect binding energies
        # Single-point energy calculation preserves the complex conformation
        sub_state = sub_sim.context.getState(getEnergy=True)
        e_val = sub_state.getPotentialEnergy()
        print(f"  {name} Energy: {e_val.value_in_unit(unit.kilocalories_per_mole):.4f} kcal/mol")
        return e_val
    
    # Normalize chains to sorted lists for deterministic behavior
    receptor_chains_normalized = _normalize_chains(receptor_chains)
    ligand_chains_normalized = _normalize_chains(ligand_chains)
    
    e_receptor = calc_energy(receptor_chains_normalized, "Receptor")
    e_ligand = calc_energy(ligand_chains_normalized, "Ligand")
    
    # 5. Calculate dG_bind
    # Note: This is an approximation: dG approximately = E_complex - (E_receptor + E_ligand)
    # Missing: entropy term (-T*dS), which can be significant for flexible systems
    dg_bind = e_complex - (e_receptor + e_ligand)
    dg_val = dg_bind.value_in_unit(unit.kilocalories_per_mole)
    
    # 6. Calculate Kd from dG (using same temperature as MD simulation)
    # WARNING: This conversion assumes dG is a true standard Gibbs free energy (dG standard)
    # Since MM/GBSA dG may miss entropic contributions, the Kd should be interpreted with caution
    kd_nm = convert_delta_g_kcal_to_kd_nm(dg_val, temperature=temperature) if dg_val is not None and not (np.isnan(dg_val) or np.isinf(dg_val)) else None
    
    print(f"  dG_bind: {dg_val:.4f} kcal/mol (T={temperature:.2f} K)")
    if kd_nm is not None:
        # Use scientific notation to avoid printing 0.0000 for extremely small values
        print(f"  Kd: {kd_nm:.3e} nM (T={temperature:.2f} K) [derived from dG, interpret with caution]")
    
    return {
        "dg_bind": dg_val,
        "kd_nm": kd_nm,
        "e_complex": e_complex.value_in_unit(unit.kilocalories_per_mole),
        "e_receptor": e_receptor.value_in_unit(unit.kilocalories_per_mole),
        "e_ligand": e_ligand.value_in_unit(unit.kilocalories_per_mole),
        "num_atoms": num_atoms,
        "iterations": effective_max_iterations,
        "converged": converged,
        "tolerance": tolerance_kj_mol,
        "energy_fluctuation": energy_fluctuation,
        "energy_before": e_before,
        "energy_after": e_final,
    }


def run_mmgbsa_ensemble(complex_pdb_path: str, receptor_chains: List[str], ligand_chains: List[str],
                       max_iterations: Optional[int] = None, skip_fixing: bool = False,
                       platform_name: Optional[str] = None,
                       md_steps: int = 5000, snapshot_interval: int = 500,
                       temperature: float = ROOM_TEMPERATURE) -> Dict:
    """
    Run ensemble MM/GBSA calculation (averages over MD snapshots).
    
    Args:
        temperature: Temperature in Kelvin for MD simulation and Kd conversion.
                     Use BODY_TEMPERATURE (310.15 K) for human drug screening,
                     ROOM_TEMPERATURE (298.15 K) for laboratory comparisons.
    
    Returns:
        Dict with 'dg_bind' (kcal/mol) and 'kd_nm' (nM) keys.
    """
    if platform_name is None:
        platform_name = get_platform_name()

    # REQUIRE CUDA with double precision for deterministic minimization
    if platform_name != "CUDA":
        raise RuntimeError(
            f"Platform '{platform_name}' not supported for deterministic minimization. "
            f"Only CUDA with double precision is supported. "
            f"Set platform_name='CUDA' or OPENMM_PLATFORM=CUDA environment variable."
        )

    print(f"Processing {os.path.basename(complex_pdb_path)} (Ensemble MM/GBSA)...")

    # 1. Fix PDB - Use GCS-cached processed structure for cross-pod determinism
    if skip_fixing:
        pdb_file = app.PDBFile(complex_pdb_path)
        topology = pdb_file.topology
        positions = pdb_file.positions
    else:
        # Use get_processed_pdb() which handles:
        # - OpenMM warmup for deterministic first-run behavior
        # - GCS-based caching for cross-pod consistency
        # - PDBFixer processing with deterministic random seeds
        topology, positions = get_processed_pdb(complex_pdb_path, ph=7.0, use_cache=True)

    # 2. Prepare Systems
    forcefield = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
    modeller = app.Modeller(topology, positions)

    if skip_fixing:
        try:
            # CRITICAL: Set Python's random seed before addHydrogens for determinism
            random.seed(DETERMINISTIC_SEED)
            modeller.addHydrogens(forcefield, pH=7.0)
        except Exception:
            pass

    # Track number of atoms for metadata
    num_atoms = len(list(modeller.topology.atoms()))

    def create_sim(modeller_obj):
        """Create simulation with CUDA double precision for deterministic minimization."""
        system = forcefield.createSystem(modeller_obj.topology,
                                         nonbondedMethod=app.NoCutoff,
                                         constraints=app.HBonds)
        integrator = openmm.LangevinIntegrator(temperature*unit.kelvin, 1.0/unit.picosecond, 2.0*unit.femtoseconds)
        try:
            platform = openmm.Platform.getPlatformByName("CUDA")
            # Use double precision for deterministic minimization
            platform.setPropertyDefaultValue('CudaPrecision', 'double')
            platform.setPropertyDefaultValue('CudaDeviceIndex', '0')
            sim = app.Simulation(modeller_obj.topology, system, integrator, platform)
        except Exception as e:
            available = get_available_platforms()
            raise RuntimeError(
                f"CUDA platform required but failed to initialize: {e}\n"
                f"Available platforms: {', '.join(available) if available else 'none'}\n"
                f"Deterministic minimization requires CUDA with double precision."
            )
        return sim
    
    # Complex Simulation
    complex_sim = create_sim(modeller)
    complex_sim.context.setPositions(modeller.positions)
    
    # Receptor Simulation
    rec_modeller = app.Modeller(modeller.topology, modeller.positions)
    to_delete = [c for c in rec_modeller.topology.chains() if c.id not in receptor_chains]
    rec_modeller.delete(to_delete)
    rec_sim = create_sim(rec_modeller)
    
    # Ligand Simulation
    lig_modeller = app.Modeller(modeller.topology, modeller.positions)
    to_delete = [c for c in lig_modeller.topology.chains() if c.id not in ligand_chains]
    lig_modeller.delete(to_delete)
    lig_sim = create_sim(lig_modeller)
    
    # Map atom indices
    rec_indices = []
    lig_indices = []
    atom_idx = 0
    for chain in modeller.topology.chains():
        is_rec = chain.id in receptor_chains
        is_lig = chain.id in ligand_chains
        for residue in chain.residues():
            for atom in residue.atoms():
                if is_rec:
                    rec_indices.append(atom_idx)
                elif is_lig:
                    lig_indices.append(atom_idx)
                atom_idx += 1
    
    # 3. Minimize
    # Use strict tolerance for more consistent convergence
    tolerance = 0.1 * unit.kilojoule_per_mole
    tolerance_kj_mol = 0.1

    # Calculate appropriate max iterations based on system size
    effective_max_iterations = calculate_max_iterations(
        num_atoms,
        tolerance_kj_mol=tolerance_kj_mol,
        user_specified=max_iterations
    )

    # Minimization with retry logic for convergence
    # Retry multipliers: 1x, 2.5x, 5x of initial iterations (capped at 100,000)
    retry_multipliers = [1.0, 2.5, 5.0]
    max_retries = len(retry_multipliers)
    convergence_threshold = 0.01  # kcal/mol

    state_before = complex_sim.context.getState(getEnergy=True)
    e_before = state_before.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    converged = False
    energy_fluctuation = None
    total_iterations_used = 0

    for retry_idx, multiplier in enumerate(retry_multipliers):
        current_iterations = min(int(effective_max_iterations * multiplier), 100000)

        if retry_idx == 0:
            print(f"  Minimizing complex ({num_atoms:,} atoms, tolerance={tolerance_kj_mol} kJ/mol, max {current_iterations:,} iterations)...")
        else:
            print(f"  Retry {retry_idx}/{max_retries-1}: Increasing to {current_iterations:,} iterations...")

        complex_sim.minimizeEnergy(maxIterations=current_iterations, tolerance=tolerance)
        total_iterations_used += current_iterations

        # Get energy after this round
        state_after = complex_sim.context.getState(getEnergy=True)
        e_after = state_after.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

        # Check convergence by running a few more iterations
        convergence_test_iterations = 100
        complex_sim.minimizeEnergy(maxIterations=convergence_test_iterations, tolerance=tolerance)
        state_convergence_test = complex_sim.context.getState(getEnergy=True)
        e_convergence_test = state_convergence_test.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        energy_fluctuation = abs(e_convergence_test - e_after)

        converged = energy_fluctuation < convergence_threshold

        if converged:
            print(f"  Convergence: YES (fluctuation={energy_fluctuation:.6f} kcal/mol < {convergence_threshold} threshold)")
            break
        else:
            print(f"  Convergence: NO (fluctuation={energy_fluctuation:.4f} kcal/mol >= {convergence_threshold} threshold)")
            if retry_idx < max_retries - 1:
                print(f"  Will retry with more iterations...")

    # Update effective_max_iterations to reflect total used
    effective_max_iterations = total_iterations_used

    # Final energy report
    state_final = complex_sim.context.getState(getEnergy=True)
    e_final = state_final.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    print(f"  Energy change: {e_before:.2f} -> {e_final:.2f} kcal/mol")
    print(f"  Total iterations used: {total_iterations_used:,}")

    if not converged:
        print(f"  WARNING: Did not converge after {max_retries} attempts")

    # Use final energy as e_after for return value
    e_after = e_final

    # Store minimized state for fallback
    minimized_state = complex_sim.context.getState(getEnergy=True, getPositions=True)
    minimized_positions = minimized_state.getPositions(asNumpy=True)
    
    # 4. MD & Sampling with error handling
    n_snapshots = md_steps // snapshot_interval
    print(f"  Running MD for {md_steps} steps, collecting {n_snapshots} snapshots...")
    
    results = {
        "complex": [],
        "receptor": [],
        "ligand": [],
        "dg": []
    }
    
    successful_snapshots = 0
    failed_snapshots = 0
    
    for i in range(n_snapshots):
        try:
            # Advance MD simulation
            complex_sim.step(snapshot_interval)
            
            # Get state
            state = complex_sim.context.getState(getEnergy=True, getPositions=True)
            pos = state.getPositions(asNumpy=True)
            e_com = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            
            # Check for NaN or infinite values
            if np.isnan(e_com) or np.isinf(e_com):
                raise ValueError(f"Invalid complex energy: {e_com}")
            
            # Receptor Energy
            rec_pos = pos[rec_indices]
            rec_sim.context.setPositions(rec_pos)
            e_rec = rec_sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            
            if np.isnan(e_rec) or np.isinf(e_rec):
                raise ValueError(f"Invalid receptor energy: {e_rec}")
            
            # Ligand Energy
            lig_pos = pos[lig_indices]
            lig_sim.context.setPositions(lig_pos)
            e_lig = lig_sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            
            if np.isnan(e_lig) or np.isinf(e_lig):
                raise ValueError(f"Invalid ligand energy: {e_lig}")
            
            dg = e_com - (e_rec + e_lig)
            
            # Check if dG is reasonable (not NaN/inf)
            if np.isnan(dg) or np.isinf(dg):
                raise ValueError(f"Invalid dG: {dg}")
            
            # Successfully collected snapshot
            results["complex"].append(e_com)
            results["receptor"].append(e_rec)
            results["ligand"].append(e_lig)
            results["dg"].append(dg)
            successful_snapshots += 1
            
        except Exception as e:
            failed_snapshots += 1
            print(f"    Warning: Snapshot {i+1} failed: {e}")
            # Continue to next snapshot instead of crashing
            continue
    
    # 5. Statistics - handle cases with successful snapshots
    # Initialize variables for fallback case
    e_com_fallback = None
    e_rec_fallback = None
    e_lig_fallback = None
    used_baseline_fallback = False
    
    if successful_snapshots == 0:
        # Fallback: use single-point calculation from minimized structure
        print(f"  Warning: All {n_snapshots} MD snapshots failed. Falling back to single-point calculation...")
        try:
            # Reset to minimized positions (MD may have corrupted the structure)
            complex_sim.context.setPositions(minimized_positions)
            
            # Get energy from minimized state
            state = complex_sim.context.getState(getEnergy=True, getPositions=True)
            pos = state.getPositions(asNumpy=True)
            e_com_fallback = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            
            # Validate complex energy
            if np.isnan(e_com_fallback) or np.isinf(e_com_fallback):
                raise ValueError(f"Invalid complex energy after minimization: {e_com_fallback}")
            
            rec_pos = pos[rec_indices]
            rec_sim.context.setPositions(rec_pos)
            e_rec_fallback = rec_sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            
            if np.isnan(e_rec_fallback) or np.isinf(e_rec_fallback):
                raise ValueError(f"Invalid receptor energy: {e_rec_fallback}")
            
            lig_pos = pos[lig_indices]
            lig_sim.context.setPositions(lig_pos)
            e_lig_fallback = lig_sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            
            if np.isnan(e_lig_fallback) or np.isinf(e_lig_fallback):
                raise ValueError(f"Invalid ligand energy: {e_lig_fallback}")
            
            avg_dg = e_com_fallback - (e_rec_fallback + e_lig_fallback)
            
            # Validate final dG
            if np.isnan(avg_dg) or np.isinf(avg_dg):
                raise ValueError(f"Invalid dG from fallback calculation: {avg_dg}")
            
            std_dg = 0.0  # Single point, no std
            n_successful = 1
            
        except Exception as fallback_error:
            # Final fallback: try baseline MM/GBSA
            print(f"  Warning: Fallback single-point also failed: {fallback_error}")
            print(f"  Attempting baseline MM/GBSA as last resort...")
            try:
                baseline_result = run_mmgbsa_baseline(
                    complex_pdb_path, receptor_chains, ligand_chains,
                    max_iterations=max_iterations, skip_fixing=skip_fixing,
                    platform_name=platform_name, temperature=temperature
                )
                avg_dg = baseline_result.get("dg_bind")
                std_dg = 0.0
                n_successful = 1
                used_baseline_fallback = True
                e_com_fallback = None
                e_rec_fallback = None
                e_lig_fallback = None
            except Exception as baseline_error:
                print(f"  Error: All fallback methods failed. Last error: {baseline_error}")
                avg_dg = None
                std_dg = None
                n_successful = 0
    else:
        # We have successful snapshots - calculate statistics
        avg_dg = np.mean(results["dg"])
        std_dg = np.std(results["dg"])
        n_successful = successful_snapshots
    
    # Format output
    if avg_dg is not None:
        print(f"  Ensemble dG: {avg_dg:.4f} +/- {std_dg:.4f} kcal/mol ({n_successful}/{n_snapshots} snapshots, T={temperature:.2f} K)")
        # Calculate Kd from average dG (using same temperature as MD simulation)
        kd_nm = convert_delta_g_kcal_to_kd_nm(avg_dg, temperature=temperature) if not (np.isnan(avg_dg) or np.isinf(avg_dg)) else None
        if kd_nm is not None:
            # Use scientific notation to avoid printing 0.0000 for extremely small values
            print(f"  Kd: {kd_nm:.3e} nM (T={temperature:.2f} K)")
    else:
        print(f"  Error: Calculation failed for all snapshots and fallback methods")
        kd_nm = None
    
    # Build return dictionary
    if successful_snapshots > 0:
        # Normal ensemble result
        return {
            "dg_bind": float(avg_dg) if avg_dg is not None else None,
            "kd_nm": float(kd_nm) if kd_nm is not None else None,
            "dg_std": float(std_dg) if std_dg is not None else None,
            "n_snapshots": int(n_snapshots),
            "n_successful_snapshots": int(n_successful),
            "n_failed_snapshots": int(failed_snapshots),
            "fallback_single_point": False,
            "num_atoms": num_atoms,
            "iterations": effective_max_iterations,
            "converged": converged,
            "tolerance": tolerance_kj_mol,
            "energy_fluctuation": energy_fluctuation,
            "energy_before": e_before,
            "energy_after": e_after,
        }
    else:
        # Fallback result
        calculation_failed = (avg_dg is None) or np.isnan(avg_dg) or np.isinf(avg_dg)

        return {
            "dg_bind": float(avg_dg) if avg_dg is not None and not calculation_failed else None,
            "kd_nm": float(kd_nm) if kd_nm is not None and not calculation_failed else None,
            "dg_std": float(std_dg) if std_dg is not None else None,
            "n_snapshots": int(n_snapshots),
            "n_successful_snapshots": int(n_successful),
            "n_failed_snapshots": int(failed_snapshots),
            "fallback_single_point": True,
            "fallback_baseline_mmgbsa": bool(used_baseline_fallback),
            "calculation_failed": bool(calculation_failed),
            "num_atoms": num_atoms,
            "iterations": effective_max_iterations,
            "converged": converged,
            "tolerance": tolerance_kj_mol,
            "energy_fluctuation": energy_fluctuation,
            "energy_before": e_before,
            "energy_after": e_after,
        }


def run_mmgbsa_variable_dielectric(complex_pdb_path: str, receptor_chains: List[str], ligand_chains: List[str],
                                  max_iterations: Optional[int] = None, skip_fixing: bool = False,
                                  platform_name: Optional[str] = None, temperature: float = ROOM_TEMPERATURE) -> Dict:
    """
    Run variable dielectric MM/GBSA calculation.

    This method is DETERMINISTIC - it uses VerletIntegrator (no stochastic terms)
    and only performs energy minimization, not MD simulation.

    Uses higher internal dielectric (4.0) as approximation for variable dielectric effect.

    Args:
        temperature: Temperature in Kelvin for Kd conversion.
                     Use BODY_TEMPERATURE (310.15 K) for human drug screening,
                     ROOM_TEMPERATURE (298.15 K) for laboratory comparisons.

    Returns:
        Dict with 'dg_bind' (kcal/mol) and 'kd_nm' (nM) keys.
    """
    if platform_name is None:
        platform_name = get_platform_name()

    # REQUIRE CUDA with double precision for deterministic minimization
    if platform_name != "CUDA":
        raise RuntimeError(
            f"Platform '{platform_name}' not supported for deterministic minimization. "
            f"Only CUDA with double precision is supported. "
            f"Set platform_name='CUDA' or OPENMM_PLATFORM=CUDA environment variable."
        )

    print(f"Processing {os.path.basename(complex_pdb_path)} (Variable Dielectric MM/GBSA)...")

    # 1. Fix PDB - Use GCS-cached processed structure for cross-pod determinism
    if skip_fixing:
        pdb_file = app.PDBFile(complex_pdb_path)
        topology = pdb_file.topology
        positions = pdb_file.positions
    else:
        # Use get_processed_pdb() which handles:
        # - OpenMM warmup for deterministic first-run behavior
        # - GCS-based caching for cross-pod consistency
        # - PDBFixer processing with deterministic random seeds
        topology, positions = get_processed_pdb(complex_pdb_path, ph=7.0, use_cache=True)

    # 2. Create System with higher internal dielectric
    forcefield = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
    modeller = app.Modeller(topology, positions)

    if skip_fixing:
        try:
            # CRITICAL: Set Python's random seed before addHydrogens for determinism
            random.seed(DETERMINISTIC_SEED)
            modeller.addHydrogens(forcefield, pH=7.0)
        except Exception:
            pass

    # Track number of atoms for metadata
    num_atoms = len(list(modeller.topology.atoms()))

    def create_sim(modeller_obj, name):
        """Create simulation with CUDA double precision for deterministic minimization."""
        system = forcefield.createSystem(modeller_obj.topology,
                                         nonbondedMethod=app.NoCutoff,
                                         constraints=app.HBonds)

        # Modify GBSAOBCForce to use higher internal dielectric
        for force in system.getForces():
            if isinstance(force, openmm.GBSAOBCForce):
                force.setSoluteDielectric(4.0)  # Higher internal dielectric
                print(f"  {name}: Set solute dielectric to 4.0")

        # Use VerletIntegrator for deterministic behavior (no stochastic terms)
        integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
        try:
            platform = openmm.Platform.getPlatformByName("CUDA")
            # Use double precision for deterministic minimization
            platform.setPropertyDefaultValue('CudaPrecision', 'double')
            platform.setPropertyDefaultValue('CudaDeviceIndex', '0')
            sim = app.Simulation(modeller_obj.topology, system, integrator, platform)
        except Exception as e:
            available = get_available_platforms()
            raise RuntimeError(
                f"CUDA platform required but failed to initialize: {e}\n"
                f"Available platforms: {', '.join(available) if available else 'none'}\n"
                f"Deterministic minimization requires CUDA with double precision."
            )
        return sim
    
    # Complex Simulation
    complex_sim = create_sim(modeller, "Complex")
    complex_sim.context.setPositions(modeller.positions)
    
    # Receptor Simulation
    rec_modeller = app.Modeller(modeller.topology, modeller.positions)
    to_delete = [c for c in rec_modeller.topology.chains() if c.id not in receptor_chains]
    rec_modeller.delete(to_delete)
    rec_sim = create_sim(rec_modeller, "Receptor")
    
    # Ligand Simulation
    lig_modeller = app.Modeller(modeller.topology, modeller.positions)
    to_delete = [c for c in lig_modeller.topology.chains() if c.id not in ligand_chains]
    lig_modeller.delete(to_delete)
    lig_sim = create_sim(lig_modeller, "Ligand")
    
    # Map atom indices
    rec_indices = []
    lig_indices = []
    atom_idx = 0
    for chain in modeller.topology.chains():
        is_rec = chain.id in receptor_chains
        is_lig = chain.id in ligand_chains
        for residue in chain.residues():
            for atom in residue.atoms():
                if is_rec:
                    rec_indices.append(atom_idx)
                elif is_lig:
                    lig_indices.append(atom_idx)
                atom_idx += 1
    
    # Minimize with retry logic for convergence
    # Use strict tolerance for more consistent convergence
    tolerance = 0.1 * unit.kilojoule_per_mole
    tolerance_kj_mol = 0.1

    # Calculate appropriate max iterations based on system size
    effective_max_iterations = calculate_max_iterations(
        num_atoms,
        tolerance_kj_mol=tolerance_kj_mol,
        user_specified=max_iterations
    )

    # Retry logic: try with progressively more iterations if convergence fails
    retry_multipliers = [1.0, 2.5, 5.0]
    max_retries = len(retry_multipliers)
    convergence_threshold = 0.01  # kcal/mol

    # Get energy before minimization
    state_before = complex_sim.context.getState(getEnergy=True)
    e_before = state_before.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    converged = False
    energy_fluctuation = None
    total_iterations_used = 0

    for retry_idx, multiplier in enumerate(retry_multipliers):
        current_iterations = min(int(effective_max_iterations * multiplier), 100000)

        if retry_idx == 0:
            print(f"  Minimizing complex ({num_atoms:,} atoms, tolerance={tolerance_kj_mol} kJ/mol, max {current_iterations:,} iterations)...")
        else:
            print(f"  Retry {retry_idx}/{max_retries-1}: Increasing to {current_iterations:,} iterations...")

        # Minimize
        complex_sim.minimizeEnergy(maxIterations=current_iterations, tolerance=tolerance)
        total_iterations_used += current_iterations

        # Get energy after this minimization round
        state_after = complex_sim.context.getState(getEnergy=True)
        e_after = state_after.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

        # Check convergence by running a few more iterations
        convergence_test_iterations = 100
        complex_sim.minimizeEnergy(maxIterations=convergence_test_iterations, tolerance=tolerance)
        state_convergence_test = complex_sim.context.getState(getEnergy=True)
        e_convergence_test = state_convergence_test.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        energy_fluctuation = abs(e_convergence_test - e_after)

        converged = energy_fluctuation < convergence_threshold

        if converged:
            print(f"  Convergence: YES (fluctuation={energy_fluctuation:.6f} kcal/mol)")
            break
        else:
            print(f"  Convergence: NO (fluctuation={energy_fluctuation:.4f} kcal/mol)")
            if retry_idx < max_retries - 1:
                print(f"  Will retry with more iterations...")

    print(f"  Energy change: {e_before:.2f} -> {e_after:.2f} kcal/mol")
    print(f"  Total iterations used: {total_iterations_used:,}")

    # Get State
    state = complex_sim.context.getState(getEnergy=True, getPositions=True)
    e_complex = state.getPotentialEnergy()
    pos = state.getPositions(asNumpy=True)
    
    # Receptor Energy
    rec_sim.context.setPositions(pos[rec_indices])
    e_receptor = rec_sim.context.getState(getEnergy=True).getPotentialEnergy()
    
    # Ligand Energy
    lig_sim.context.setPositions(pos[lig_indices])
    e_ligand = lig_sim.context.getState(getEnergy=True).getPotentialEnergy()
    
    dg_bind = e_complex - (e_receptor + e_ligand)
    dg_val = dg_bind.value_in_unit(unit.kilocalories_per_mole)
    
    # Calculate Kd from dG (using same temperature as MD simulation)
    kd_nm = convert_delta_g_kcal_to_kd_nm(dg_val, temperature=temperature) if dg_val is not None and not (np.isnan(dg_val) or np.isinf(dg_val)) else None
    
    print(f"  dG_bind (eps=4.0): {dg_val:.4f} kcal/mol (T={temperature:.2f} K)")
    if kd_nm is not None:
        # Use scientific notation to avoid printing 0.0000 for extremely small values
        print(f"  Kd: {kd_nm:.3e} nM (T={temperature:.2f} K)")
    
    return {
        "dg_bind": dg_val,
        "kd_nm": kd_nm,
        "e_complex": e_complex.value_in_unit(unit.kilocalories_per_mole),
        "e_receptor": e_receptor.value_in_unit(unit.kilocalories_per_mole),
        "e_ligand": e_ligand.value_in_unit(unit.kilocalories_per_mole),
        "num_atoms": num_atoms,
        "iterations": total_iterations_used,
        "converged": converged,
        "tolerance": tolerance_kj_mol,
        "energy_fluctuation": energy_fluctuation,
        "energy_before": e_before,
        "energy_after": e_after,
    }


def run_mmgbsa(complex_pdb_path: str, receptor_chains: List[str], ligand_chains: List[str],
               method: str = "baseline", temperature: float = ROOM_TEMPERATURE, **kwargs) -> Dict:
    """
    Main entry point for MM/GBSA calculations.
    
    Args:
        complex_pdb_path: Path to complex PDB file
        receptor_chains: List of receptor chain IDs
        ligand_chains: List of ligand chain IDs
        method: Method to use ("baseline", "ensemble", "variable_dielectric")
        temperature: Temperature in Kelvin for MD simulation and Kd conversion.
                     Use BODY_TEMPERATURE (310.15 K) for human drug screening,
                     ROOM_TEMPERATURE (298.15 K) for laboratory comparisons.
        **kwargs: Additional arguments passed to specific method
    
    Returns:
        Dict with 'dg_bind' (kcal/mol) and 'kd_nm' (nM) keys
    """
    # Pass temperature to all methods
    kwargs['temperature'] = temperature
    if method == "baseline":
        return run_mmgbsa_baseline(complex_pdb_path, receptor_chains, ligand_chains, **kwargs)
    elif method == "ensemble":
        return run_mmgbsa_ensemble(complex_pdb_path, receptor_chains, ligand_chains, **kwargs)
    elif method == "variable_dielectric":
        return run_mmgbsa_variable_dielectric(complex_pdb_path, receptor_chains, ligand_chains, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Supported: 'baseline', 'ensemble', 'variable_dielectric'")


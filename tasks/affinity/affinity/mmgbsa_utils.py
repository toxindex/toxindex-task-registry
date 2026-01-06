"""MM/GBSA utilities for binding affinity calculations."""

import os
import json
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import openmm
from openmm import unit
from openmm import app
import pdbfixer
from affinity.affinity_utils import convert_delta_g_kcal_to_kd_nm, BODY_TEMPERATURE, ROOM_TEMPERATURE

# Set deterministic random seed for reproducibility
# This ensures energy minimization converges to the same state
DETERMINISTIC_SEED = 42


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
    fixer.addMissingAtoms()
    
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
                       max_iterations: int = 1000, skip_fixing: bool = False,
                       platform_name: Optional[str] = None, temperature: float = BODY_TEMPERATURE) -> Dict:
    """
    Run baseline MM/GBSA calculation (single-trajectory method).

    This method is DETERMINISTIC - it uses VerletIntegrator (no stochastic terms)
    and only performs energy minimization, not MD simulation.
    
    DETERMINISM GUARANTEES:
    - Chains are normalized (sorted) to ensure consistent ordering
    - Random seed is set for reproducible minimization
    - Single-trajectory method ensures receptor/ligand use same coordinates as complex
    
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
    
    # 1. Fix PDB
    if skip_fixing:
        pdb_file = app.PDBFile(complex_pdb_path)
        topology = pdb_file.topology
        positions = pdb_file.positions
    else:
        fixer = pdbfixer.PDBFixer(filename=complex_pdb_path)
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.0)
        topology = fixer.topology
        positions = fixer.positions
    
    # 2. Create System
    forcefield = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
    modeller = app.Modeller(topology, positions)
    
    if skip_fixing:
        try:
            modeller.addHydrogens(forcefield, pH=7.0)
        except Exception:
            pass
    
    constraints = None if skip_fixing else app.HBonds
    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.NoCutoff,
                                     constraints=constraints)
    
    # 3. Minimize Complex
    # Use VerletIntegrator for deterministic behavior (no stochastic terms)
    # Langevin integrator is unnecessary since we only do minimization, not MD
    integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
    
    try:
        platform = openmm.Platform.getPlatformByName(platform_name)
        simulation = app.Simulation(modeller.topology, system, integrator, platform)
        print(f"  Using OpenMM Platform: {platform.getName()}")
    except Exception as e:
        # Platform not available, try to get best available platform
        available = get_available_platforms()
        if available:
            fallback_platform = available[0]  # Use first available
            print(f"  Warning: Platform '{platform_name}' not available (Error: {e})")
            print(f"  Available platforms: {', '.join(available)}")
            print(f"  Falling back to: {fallback_platform}")
            try:
                platform = openmm.Platform.getPlatformByName(fallback_platform)
                simulation = app.Simulation(modeller.topology, system, integrator, platform)
                print(f"  Using OpenMM Platform: {platform.getName()}")
            except Exception:
                # Last resort: use default
                print(f"  Warning: Could not set platform, using default")
                simulation = app.Simulation(modeller.topology, system, integrator)
        else:
            print(f"  Warning: Could not set platform {platform_name}, using default. Error: {e}")
            simulation = app.Simulation(modeller.topology, system, integrator)
    
    # Set positions - VerletIntegrator is deterministic (no stochastic terms)
    # For deterministic behavior, ensure:
    # 1. Chain ordering is normalized (done via _normalize_chains)
    # 2. Same platform is used
    # 3. Same initial positions
    simulation.context.setPositions(modeller.positions)
    
    print(f"  Minimizing complex (max {max_iterations} iterations)...")
    # Check convergence by comparing energy before and after minimization
    state_before = simulation.context.getState(getEnergy=True)
    e_before = state_before.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    
    simulation.minimizeEnergy(maxIterations=max_iterations)
    
    # Verify minimization converged
    state_after = simulation.context.getState(getEnergy=True)
    e_after = state_after.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    energy_change = e_after - e_before
    if abs(energy_change) > 1000:  # Large energy change suggests issues
        print(f"  WARNING: Large energy change during minimization: {energy_change:.2f} kcal/mol")
        print(f"  This may indicate convergence issues or system problems.")
    
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    e_complex = state.getPotentialEnergy()
    minimized_positions = state.getPositions()
    
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
        sub_integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
        
        try:
            platform = openmm.Platform.getPlatformByName(platform_name)
            sub_sim = app.Simulation(sub_modeller.topology, sub_system, sub_integrator, platform)
        except Exception:
            sub_sim = app.Simulation(sub_modeller.topology, sub_system, sub_integrator)
        
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
    }


def run_mmgbsa_ensemble(complex_pdb_path: str, receptor_chains: List[str], ligand_chains: List[str],
                       max_iterations: int = 100, skip_fixing: bool = False,
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
    
    print(f"Processing {os.path.basename(complex_pdb_path)} (Ensemble MM/GBSA)...")
    
    # 1. Fix PDB
    if skip_fixing:
        pdb_file = app.PDBFile(complex_pdb_path)
        topology = pdb_file.topology
        positions = pdb_file.positions
    else:
        fixer = pdbfixer.PDBFixer(filename=complex_pdb_path)
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.0)
        topology = fixer.topology
        positions = fixer.positions
    
    # 2. Prepare Systems
    forcefield = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
    modeller = app.Modeller(topology, positions)
    
    if skip_fixing:
        try:
            modeller.addHydrogens(forcefield, pH=7.0)
        except Exception:
            pass
    
    def create_sim(modeller_obj):
        system = forcefield.createSystem(modeller_obj.topology,
                                         nonbondedMethod=app.NoCutoff,
                                         constraints=app.HBonds)
        integrator = openmm.LangevinIntegrator(temperature*unit.kelvin, 1.0/unit.picosecond, 2.0*unit.femtoseconds)
        try:
            platform = openmm.Platform.getPlatformByName(platform_name)
            sim = app.Simulation(modeller_obj.topology, system, integrator, platform)
        except Exception:
            sim = app.Simulation(modeller_obj.topology, system, integrator)
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
    print(f"  Minimizing complex (max {max_iterations} iterations)...")
    complex_sim.minimizeEnergy(maxIterations=max_iterations)
    
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
            "fallback_single_point": False
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
            "calculation_failed": bool(calculation_failed)
        }


def run_mmgbsa_variable_dielectric(complex_pdb_path: str, receptor_chains: List[str], ligand_chains: List[str],
                                  max_iterations: int = 100, skip_fixing: bool = False,
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
    
    print(f"Processing {os.path.basename(complex_pdb_path)} (Variable Dielectric MM/GBSA)...")
    
    # 1. Fix PDB
    if skip_fixing:
        pdb_file = app.PDBFile(complex_pdb_path)
        topology = pdb_file.topology
        positions = pdb_file.positions
    else:
        fixer = pdbfixer.PDBFixer(filename=complex_pdb_path)
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.0)
        topology = fixer.topology
        positions = fixer.positions
    
    # 2. Create System with higher internal dielectric
    forcefield = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
    modeller = app.Modeller(topology, positions)
    
    if skip_fixing:
        try:
            modeller.addHydrogens(forcefield, pH=7.0)
        except Exception:
            pass
    
    def create_sim(modeller_obj, name):
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
            platform = openmm.Platform.getPlatformByName(platform_name)
            sim = app.Simulation(modeller_obj.topology, system, integrator, platform)
        except Exception:
            sim = app.Simulation(modeller_obj.topology, system, integrator)
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
    
    # Minimize
    print(f"  Minimizing complex (max {max_iterations} iterations)...")
    complex_sim.minimizeEnergy(maxIterations=max_iterations)
    
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


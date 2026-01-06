"""
Calculate recommended max iterations for energy minimization based on system size.

For our three example PDB files:
- 1S78_complex.pdb: 7599 atoms
- 2DD8_complex.pdb: 4740 atoms  
- 2FJG_complex.pdb: 4781 atoms
"""

def calculate_max_iterations(num_atoms: int, tolerance_kj_mol: float = 0.001) -> int:
    """
    Calculate appropriate max iterations based on system size and tolerance.
    
    Rules:
    - Base: 1 iteration per atom (minimum 1000)
    - For very strict tolerance (0.001 kJ/mol): multiply by 5-10
    - For strict tolerance (0.01 kJ/mol): multiply by 3
    - For moderate tolerance (0.1 kJ/mol): multiply by 2
    - Safety limit: 20000 iterations maximum
    
    Args:
        num_atoms: Number of atoms in system
        tolerance_kj_mol: Tolerance in kJ/mol
    
    Returns:
        Recommended max iterations
    """
    # Base iterations: ~1 iteration per atom, minimum 1000
    base_iterations = max(1000, num_atoms)
    
    # Adjust for tolerance strictness
    if tolerance_kj_mol <= 0.001:
        # Very strict: need 5-10x more iterations
        multiplier = 7  # Middle ground between 5 and 10
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
    max_iterations = min(calculated, 20000)
    
    return max_iterations


# Calculate for our example files
if __name__ == "__main__":
    pdb_sizes = {
        "1S78": 7599,
        "2DD8": 4740,
        "2FJG": 4781,
    }
    
    tolerance = 0.001  # Very strict tolerance (kJ/mol)
    
    print("=" * 60)
    print("Recommended Max Iterations for Energy Minimization")
    print("=" * 60)
    print(f"Tolerance: {tolerance} kJ/mol (very strict)")
    print(f"Rule: Base iterations = max(1000, num_atoms) × 7")
    print()
    
    for pdb_id, num_atoms in pdb_sizes.items():
        base = max(1000, num_atoms)
        recommended = calculate_max_iterations(num_atoms, tolerance)
        
        print(f"{pdb_id}:")
        print(f"  Atoms: {num_atoms:,}")
        print(f"  Base iterations: {base:,} (max(1000, {num_atoms}))")
        print(f"  Multiplier: 7 (for tolerance {tolerance} kJ/mol)")
        print(f"  Calculated: {base * 7:,}")
        print(f"  Recommended: {recommended:,} iterations")
        print(f"  Current fixed: 5,000 iterations")
        print(f"  Difference: {recommended - 5000:+,} iterations")
        print()


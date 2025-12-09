"""Binding affinity calculation utilities."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# Constants
# Standard gas constant R = 8.314 J/(mol·K)
R_J = 8.314  # J/(mol·K) - Standard SI units
R = 8.314e-3  # kJ/(mol·K) - For backward compatibility (deprecated, use R_J)

# Temperature constants (Kelvin)
ROOM_TEMPERATURE = 298.15  # 25°C - Standard laboratory temperature
BODY_TEMPERATURE = 310.15  # 37°C - Human body temperature
DEFAULT_TEMPERATURE = BODY_TEMPERATURE  # Default for backward compatibility


def convert_ic50_to_pic50(ic50: float) -> float:
    """
    Convert IC50 to pIC50.
    
    pIC50 = -log10(IC50)
    Higher pIC50 = stronger activity
    """
    return -np.log10(ic50 + 1e-10)  # Add small epsilon to avoid log(0)


def convert_pic50_to_ic50(pic50: float) -> float:
    """Convert pIC50 to IC50."""
    return 10 ** (-pic50)


def convert_kd_to_delta_g(kd: float, temperature: float = DEFAULT_TEMPERATURE) -> float:
    """
    Convert dissociation constant Kd to binding free energy ΔG.
    
    Formula: ΔG = -RT ln(Kd)
    More negative ΔG = stronger binding
    
    Args:
        kd: Dissociation constant in M (molar)
        temperature: Temperature in Kelvin
    
    Returns:
        Binding free energy ΔG in kJ/mol
    """
    # Use standard R = 8.314 J/(mol·K), result will be in J/mol
    delta_g_joule = -R_J * temperature * np.log(kd + 1e-10)
    # Convert from J/mol to kJ/mol
    return delta_g_joule / 1000


def convert_delta_g_to_kd(delta_g: float, temperature: float = DEFAULT_TEMPERATURE) -> float:
    """
    Convert binding free energy ΔG to Kd.
    
    Formula: Kd = exp(-ΔG / (RT))
    
    Args:
        delta_g: Binding free energy in kJ/mol
        temperature: Temperature in Kelvin
    
    Returns:
        Kd in M (molar)
    """
    # Convert ΔG from kJ/mol to J/mol to match standard R = 8.314 J/(mol·K)
    delta_g_joule = delta_g * 1000  # kJ/mol -> J/mol
    return np.exp(-delta_g_joule / (R_J * temperature))


def convert_delta_g_kcal_to_kd_nm(delta_g_kcal: float, temperature: float = DEFAULT_TEMPERATURE) -> float:
    """
    Convert binding free energy ΔG (in kcal/mol) to Kd (in nM).
    
    This is the standard conversion for MM/GBSA results which return ΔG in kcal/mol.
    
    Formula: Kd = exp(-ΔG / (RT)) where R = 0.001987 kcal/(mol·K)
    
    IMPORTANT THERMODYNAMIC NOTES:
    - This conversion assumes ΔG is the standard Gibbs free energy change (ΔG°)
    - MM/GBSA estimates may not include full entropic contributions
    - Temperature significantly affects Kd: ~2-3x change per 10°C
    - For human drug screening, use BODY_TEMPERATURE (310.15 K / 37°C)
    - For laboratory comparisons, use ROOM_TEMPERATURE (298.15 K / 25°C)
    
    CONVERSION ACCURACY LIMITATIONS:
    - The exponential relationship makes Kd very sensitive to ΔG errors
    - A 0.1 kcal/mol error in ΔG → ~1.2x error in Kd
    - A 0.3 kcal/mol error in ΔG → ~1.5x error in Kd
    - A 0.5 kcal/mol error in ΔG → ~2x error in Kd
    - Temperature differences (5-10°C) can cause 2-4% additional error
    - Experimental uncertainty (±0.1-0.3 kcal/mol) leads to 2-3x uncertainty in Kd
    - For ranking tasks, consider using ΔG directly instead of Kd to avoid
      exponential amplification of errors
    
    Args:
        delta_g_kcal: Binding free energy in kcal/mol
        temperature: Temperature in Kelvin (default: 298.15 K / 25°C)
                     Use BODY_TEMPERATURE (310.15 K) for human drug screening
    
    Returns:
        Kd in nM (nanomolar)
    """
    # Gas constant in kcal/(mol·K)
    R_kcal = 0.001987  # kcal/(mol·K)
    
    # Convert ΔG (kcal/mol) to Kd (M)
    # Formula from benchmark: ΔG = -RT ln(Kd/c°) where c° = 1M
    # Rearranging: Kd = c° * exp(ΔG/(RT)) = exp(ΔG/(RT))  (c° = 1M)
    # Note: The benchmark uses ΔG = RT ln(Kd/c°) convention (positive ΔG for dissociation)
    # But reports negative values, so we use: Kd = exp(ΔG/(RT))
    kd_molar = np.exp(delta_g_kcal / (R_kcal * temperature))
    
    # Convert from M to nM (keep full precision; formatting handled at output)
    kd_nm = kd_molar * 1e9
    return kd_nm


def calculate_delta_metrics(variant_value: float, reference_value: float) -> float:
    """
    Calculate delta metric (difference between variant and reference).
    
    Used for:
    - delta_pIC50 = pIC50_variant - pIC50_reference
    - delta_delta_G = ΔG_variant - ΔG_reference
    """
    return variant_value - reference_value


def rank_variants(results_df: pd.DataFrame, 
                 metric: str = 'delta_pIC50',
                 ascending: bool = False) -> pd.DataFrame:
    """
    Rank variants by a given metric.
    
    Args:
        results_df: DataFrame with affinity predictions
        metric: Metric to rank by (delta_pIC50, delta_delta_G, pIC50, etc.)
        ascending: If True, rank ascending (lower is better)
    
    Returns:
        Ranked DataFrame with added 'rank' column
    """
    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results")
    
    ranked_df = results_df.sort_values(metric, ascending=ascending).copy()
    ranked_df['rank'] = range(1, len(ranked_df) + 1)
    
    return ranked_df


def calculate_batch_deltas(results_df: pd.DataFrame,
                          reference_name: str,
                          variant_col: str = 'name',
                          value_col: str = 'pIC50') -> pd.DataFrame:
    """
    Calculate delta metrics for a batch of variants relative to a reference.
    
    Args:
        results_df: DataFrame with predictions
        reference_name: Name of reference ligand
        variant_col: Column name containing variant identifiers
        value_col: Column name containing values to calculate deltas for
    
    Returns:
        DataFrame with added delta columns
    """
    ref_row = results_df[results_df[variant_col] == reference_name]
    if len(ref_row) == 0:
        raise ValueError(f"Reference '{reference_name}' not found in results")
    
    ref_value = ref_row[value_col].iloc[0]
    results_df[f'delta_{value_col}'] = results_df[value_col] - ref_value
    
    return results_df


def filter_by_affinity(results_df: pd.DataFrame,
                      min_pic50: Optional[float] = None,
                      max_pic50: Optional[float] = None,
                      min_delta_pic50: Optional[float] = None) -> pd.DataFrame:
    """
    Filter results by affinity thresholds.
    
    Args:
        results_df: DataFrame with predictions
        min_pic50: Minimum pIC50 threshold
        max_pic50: Maximum pIC50 threshold
        min_delta_pic50: Minimum delta_pIC50 threshold
    
    Returns:
        Filtered DataFrame
    """
    filtered_df = results_df.copy()
    
    if min_pic50 is not None and 'pIC50' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['pIC50'] >= min_pic50]
    
    if max_pic50 is not None and 'pIC50' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['pIC50'] <= max_pic50]
    
    if min_delta_pic50 is not None and 'delta_pIC50' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['delta_pIC50'] >= min_delta_pic50]
    
    return filtered_df


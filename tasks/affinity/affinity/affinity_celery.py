"""Celery task for binding affinity calculations with multiple methods."""

import os
import tempfile
import time
import pandas as pd
import numpy as np
import json
import re
import logging
from pathlib import Path
from typing import Optional, Dict, List
from scipy.stats import spearmanr, kendalltau
from workflows.celery_app import celery
from celery import group, chord
from celery.result import AsyncResult
from webserver.model.task import Task
from webserver.storage import GCSFileStorage
from webserver.model.file import File
from workflows.utils import (
    emit_status,
    download_gcs_file_to_temp,
    get_redis_connection,
    emit_task_file,
    emit_task_message,
)
from webserver.model.message import MessageSchema
from affinity.mmgbsa_utils import (
    extract_case_id,
    load_metadata,
    validate_metadata,
    run_mmgbsa,
)
from affinity.affinity_utils import BODY_TEMPERATURE, ROOM_TEMPERATURE

logger = logging.getLogger(__name__)

def create_llm():
    """Create Vertex AI Gemini LLM instance."""
    from langchain_google_vertexai import ChatVertexAI
    return ChatVertexAI(
        model_name="gemini-2.5-flash",
        temperature=0.1,
        max_output_tokens=1024,
        project="873471276793",
        location="us-east4",
    )


def extract_temperature_from_query(user_query: str) -> Optional[float]:
    """
    Extract temperature preference from user query.
    
    Recognizes:
    - "body temperature", "37°C", "37 C", "310 K", "310K", "human body temp"
    - "room temperature", "25°C", "25 C", "298 K", "298K", "room temp"
    - Explicit temperature values in Celsius or Kelvin
    
    Returns temperature in Kelvin, or None if not found.
    """
    if not user_query or not user_query.strip():
        return None
    
    query_lower = user_query.lower()
    
    # First, try simple keyword matching for common temperature references
    if any(phrase in query_lower for phrase in ["body temperature", "body temp", "human body", "37°c", "37 c", "37c"]):
        return BODY_TEMPERATURE
    
    if any(phrase in query_lower for phrase in ["room temperature", "room temp", "25°c", "25 c", "25c"]):
        return ROOM_TEMPERATURE
    
    # Try to extract explicit temperature values
    import re
    
    # Pattern for Kelvin: "310 K", "310K", "310 kelvin"
    kelvin_pattern = r'(\d+(?:\.\d+)?)\s*k(?:elvin)?\b'
    kelvin_match = re.search(kelvin_pattern, query_lower)
    if kelvin_match:
        try:
            temp_k = float(kelvin_match.group(1))
            if 200 <= temp_k <= 400:  # Reasonable range
                return temp_k
        except ValueError:
            pass
    
    # Pattern for Celsius: "37°C", "37 C", "37C", "37 degrees celsius"
    celsius_pattern = r'(\d+(?:\.\d+)?)\s*(?:°|degrees?)?\s*c(?:elsius)?\b'
    celsius_match = re.search(celsius_pattern, query_lower)
    if celsius_match:
        try:
            temp_c = float(celsius_match.group(1))
            if -50 <= temp_c <= 100:  # Reasonable range
                temp_k = temp_c + 273.15
                return temp_k
        except ValueError:
            pass
    
    # If no explicit temperature found, try LLM extraction
    try:
        llm = create_llm()
        
        prompt = f"""Extract temperature preference from the following user query for binding affinity calculation.

The user may specify:
- Body temperature (37°C / 310.15 K) - for human drug screening
- Room temperature (25°C / 298.15 K) - for laboratory comparisons
- Explicit temperature in Celsius or Kelvin

User query: "{user_query}"

If the user mentions a specific temperature, return ONLY the temperature value in Kelvin as a number (e.g., 310.15, 298.15, 300.0).
If the user mentions "body temperature" or "37", return 310.15.
If the user mentions "room temperature" or "25", return 298.15.
If no temperature is mentioned, return null.

Return ONLY the number or null, nothing else:"""

        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract number from response
        number_match = re.search(r'(\d+(?:\.\d+)?)', response_text)
        if number_match:
            try:
                temp_k = float(number_match.group(1))
                if 200 <= temp_k <= 400:  # Validate reasonable range
                    return temp_k
            except ValueError:
                pass
        
        # Check for null/None
        if 'null' in response_text.lower() or 'none' in response_text.lower():
            return None
        
        return None
        
    except Exception as e:
        print(f"Error extracting temperature from query: {e}")
        return None


def extract_methods_from_query(user_query: str) -> List[str]:
    """
    Extract method preferences from user query using Gemini 2.5 Flash.
    
    Recognizes: Baseline, Ensemble, Variable Dielectric (or VD)
    
    Returns list of method names in internal format, or empty list if none found.
    """
    if not user_query or not user_query.strip():
        return []
    
    # Method name mappings (user-friendly -> internal)
    method_mappings = {
        "baseline": "baseline",
        "ensemble": "ensemble",
        "variable dielectric": "variable_dielectric",
        "vd": "variable_dielectric",
        "variable_dielectric": "variable_dielectric",
    }
    
    # First, try simple keyword matching
    query_lower = user_query.lower()
    found_methods = []
    
    for keyword, method in method_mappings.items():
        if keyword in query_lower:
            if method not in found_methods:
                found_methods.append(method)
    
    # If found methods, return them
    if found_methods:
        return found_methods
    
    # If no methods found via keywords, use LLM to extract
    try:
        llm = create_llm()
        
        prompt = f"""Extract method preferences from the following user query for binding affinity calculation.

Available methods:
- Baseline (or baseline)
- Ensemble (or ensemble)
- Variable Dielectric (or VD, variable dielectric)

User query: "{user_query}"

If the user mentions specific methods, return ONLY a JSON array of method names in lowercase with underscores (e.g., ["baseline", "ensemble"]).
If no methods are mentioned or user wants all methods, return an empty array [].

Examples:
- "use baseline and ensemble" -> ["baseline", "ensemble"]
- "run variable dielectric" -> ["variable_dielectric"]
- "use VD" -> ["variable_dielectric"]
- "calculate binding affinity" -> []
- "use all methods" -> []

Return ONLY the JSON array, nothing else:"""

        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract JSON array from response
        json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if json_match:
            methods_json = json_match.group(0)
            methods = json.loads(methods_json)
            # Validate and normalize method names
            valid_methods = []
            for method in methods:
                method_lower = method.lower().replace(' ', '_')
                if method_lower in method_mappings.values():
                    if method_lower not in valid_methods:
                        valid_methods.append(method_lower)
            return valid_methods
        
        return []
        
    except Exception as e:
        print(f"Error extracting methods from query: {e}")
        return []




def generate_affinity_summary(comparison_df: pd.DataFrame, methods: List[str], num_pdb_files: int, temperature: float, execution_time: Optional[float] = None) -> str:
    """
    Generate a summary message for the affinity calculation results.
    
    Args:
        comparison_df: DataFrame with comparison results
        methods: List of methods that were run
        num_pdb_files: Number of PDB files processed
        temperature: Temperature used for calculations (in Kelvin)
        execution_time: Optional total execution time in seconds
    
    Returns:
        Formatted summary message string
    """
    temp_c = temperature - 273.15
    
    summary_lines = [
        f"## Binding Affinity Calculation Complete",
        f"",
        f"**Summary:**",
        f"- Processed {num_pdb_files} PDB complex file(s)",
        f"- Method(s) used: {', '.join(methods)}",
        f"- Temperature: {temperature:.2f} K ({temp_c:.1f}°C)",
        f"- Total cases analyzed: {len(comparison_df)}",
    ]
    
    # Add execution time if provided
    if execution_time is not None:
        if execution_time < 60:
            summary_lines.append(f"- Total execution time: {execution_time:.1f} seconds")
        elif execution_time < 3600:
            minutes = int(execution_time // 60)
            seconds = execution_time % 60
            summary_lines.append(f"- Total execution time: {minutes} minute(s) {seconds:.1f} seconds")
        else:
            hours = int(execution_time // 3600)
            minutes = int((execution_time % 3600) // 60)
            seconds = execution_time % 60
            summary_lines.append(f"- Total execution time: {hours} hour(s) {minutes} minute(s) {seconds:.1f} seconds")
    
    summary_lines.append(f"")
    
    # Add method-specific summaries
    for method in methods:
        method_dg_col = f'{method}_dG'
        method_kd_col = f'{method}_Kd_nM'
        logger.debug(f"[DEBUG generate_affinity_summary] Processing method={method}, looking for columns: {method_dg_col}, {method_kd_col}")
        
        if method_dg_col in comparison_df.columns:
            logger.debug(f"[DEBUG generate_affinity_summary] Found {method_dg_col} column")
            valid_dg = comparison_df[method_dg_col].dropna()
            if len(valid_dg) > 0:
                avg_dg = valid_dg.mean()
                min_dg = valid_dg.min()
                max_dg = valid_dg.max()
                summary_lines.append(f"**{method.capitalize()} Method:**")
                summary_lines.append(f"- Average ΔG: {avg_dg:.2f} kcal/mol")
                summary_lines.append(f"- Range: {min_dg:.2f} to {max_dg:.2f} kcal/mol")
                
                # Add Kd statistics if available
                if method_kd_col in comparison_df.columns:
                    logger.debug(f"[DEBUG generate_affinity_summary] Found {method_kd_col} column, processing Kd values")
                    try:
                        # Kd values are stored as strings in scientific notation, need to convert
                        valid_kd = []
                        kd_series = comparison_df[method_kd_col].dropna()
                        logger.debug(f"[DEBUG generate_affinity_summary] Processing {len(kd_series)} Kd values")
                        for kd_str in kd_series:
                            try:
                                if isinstance(kd_str, str):
                                    # Parse scientific notation like "1.234e-20"
                                    kd_val = float(kd_str)
                                else:
                                    kd_val = float(kd_str)
                                if kd_val > 0 and not np.isnan(kd_val) and not np.isinf(kd_val):
                                    valid_kd.append(kd_val)
                            except (ValueError, TypeError) as parse_err:
                                logger.debug(f"[DEBUG generate_affinity_summary] Failed to parse Kd value '{kd_str}': {parse_err}")
                                continue
                        
                        logger.debug(f"[DEBUG generate_affinity_summary] Found {len(valid_kd)} valid Kd values")
                        if len(valid_kd) > 0:
                            # Convert to numpy array for easier handling
                            kd_array = np.array(valid_kd)
                            # Use geometric mean for Kd (more appropriate for exponential values)
                            # Avoid log(0) or log(negative) errors
                            kd_array_positive = kd_array[kd_array > 0]
                            if len(kd_array_positive) > 0:
                                avg_kd = np.exp(np.mean(np.log(kd_array_positive)))
                                min_kd = kd_array.min()
                                max_kd = kd_array.max()
                                logger.debug(f"[DEBUG generate_affinity_summary] Calculated Kd stats: avg={avg_kd:.3e}, min={min_kd:.3e}, max={max_kd:.3e}")
                                
                                summary_lines.append(f"- Average Kd: {avg_kd:.3e} nM")
                                summary_lines.append(f"- Kd Range: {min_kd:.3e} to {max_kd:.3e} nM")
                                summary_lines.append(f"")
                                summary_lines.append(f"  *Note: Kd (dissociation constant) values are derived from ΔG using:*")
                                summary_lines.append(f"  *Kd = exp(ΔG / (R × T)) × 10⁹ nM*")
                                summary_lines.append(f"  *where R = 0.001987 kcal/(mol·K), T = {temperature:.2f} K, and ΔG is in kcal/mol*")
                                summary_lines.append(f"  *Since ΔG is negative for favorable binding, this yields small Kd values.*")
                                summary_lines.append(f"  *Kd should be interpreted with caution: MM/GBSA ΔG may not include*")
                                summary_lines.append(f"  *full entropic contributions, and Kd is exponentially sensitive to ΔG errors.*")
                                summary_lines.append(f"")
                            else:
                                logger.warning(f"[WARNING] No positive Kd values found for {method} after filtering")
                        else:
                            logger.warning(f"[WARNING] No valid Kd values found for {method}")
                    except Exception as e:
                        # If Kd processing fails, just skip it but continue with the rest
                        logger.error(f"[ERROR] Error processing Kd values for {method}: {e}", exc_info=True)
                        summary_lines.append(f"")
                else:
                    logger.debug(f"[DEBUG generate_affinity_summary] {method_kd_col} column not found in comparison_df")
                    summary_lines.append(f"")
    
    # Add groundtruth comparison if available
    if 'groundtruth_dG' in comparison_df.columns:
        summary_lines.append(f"**Groundtruth Comparison:**")
        summary_lines.append(f"- Experimental values available for comparison")
        summary_lines.append(f"")
        
        # Calculate ranking quality metrics for each method
        # Get valid ground truth data (non-null)
        valid_gt_mask = comparison_df['groundtruth_dG'].notna()
        if valid_gt_mask.sum() >= 2:  # Need at least 2 data points for correlation
            for method in methods:
                method_dg_col = f'{method}_dG'
                if method_dg_col not in comparison_df.columns:
                    continue
                
                # Get valid predictions (non-null) that also have ground truth
                valid_pred_mask = comparison_df[method_dg_col].notna() & valid_gt_mask
                valid_data = comparison_df.loc[valid_pred_mask]
                
                if len(valid_data) < 2:
                    continue
                
                # Extract values for correlation calculations
                gt_values = valid_data['groundtruth_dG'].values
                pred_values = valid_data[method_dg_col].values
                
                # Calculate Spearman rank correlation
                try:
                    spearman_corr, spearman_p = spearmanr(gt_values, pred_values)
                except Exception as e:
                    logger.warning(f"[WARNING] Failed to calculate Spearman correlation for {method}: {e}")
                    spearman_corr, spearman_p = np.nan, np.nan
                
                # Calculate Kendall's tau
                try:
                    kendall_tau, kendall_p = kendalltau(gt_values, pred_values)
                except Exception as e:
                    logger.warning(f"[WARNING] Failed to calculate Kendall's tau for {method}: {e}")
                    kendall_tau, kendall_p = np.nan, np.nan
                
                # Calculate Top 10 accuracy
                top10_acc = np.nan
                if len(valid_data) >= 10:
                    try:
                        # Sort by ground truth (ascending = strongest first)
                        gt_sorted = valid_data.sort_values('groundtruth_dG', ascending=True)
                        top10_gt_cases = set(gt_sorted['Case'].head(10).values)
                        
                        # Sort by predicted (ascending = strongest first)
                        pred_sorted = valid_data.sort_values(method_dg_col, ascending=True)
                        top10_pred_cases = set(pred_sorted['Case'].head(10).values)
                        
                        # Calculate intersection
                        top10_acc = len(top10_gt_cases & top10_pred_cases) / 10.0
                    except Exception as e:
                        logger.warning(f"[WARNING] Failed to calculate Top 10 accuracy for {method}: {e}")
                
                # Add metrics to summary
                summary_lines.append(f"**{method.capitalize()} Ranking Quality:**")
                if not np.isnan(top10_acc):
                    summary_lines.append(f"- Top 10 Accuracy: {top10_acc:.1%} ({int(top10_acc * 10)}/10)")
                if not np.isnan(spearman_corr):
                    summary_lines.append(f"- Spearman Rank Correlation: {spearman_corr:.3f} (p={spearman_p:.3e})" if not np.isnan(spearman_p) else f"- Spearman Rank Correlation: {spearman_corr:.3f}")
                if not np.isnan(kendall_tau):
                    summary_lines.append(f"- Kendall's Tau: {kendall_tau:.3f} (p={kendall_p:.3e})" if not np.isnan(kendall_p) else f"- Kendall's Tau: {kendall_tau:.3f}")
                summary_lines.append(f"")
        else:
            summary_lines.append(f"- Insufficient ground truth data for ranking quality metrics (need at least 2 cases)")
            summary_lines.append(f"")
    
    summary_lines.append(f"Detailed results are available in the comparison table CSV file.")
    
    result = "\n".join(summary_lines)
    logger.debug(f"[DEBUG generate_affinity_summary] Summary message generated, length: {len(result)} characters")
    return result


def generate_comparison_table(results_by_method: dict, groundtruth_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Generate comprehensive comparison table with all methods and rankings.
    
    Args:
        results_by_method: Dict mapping method names to DataFrames with Case and Predicted_dG columns
        groundtruth_df: Optional DataFrame with Case and Experimental_dG columns
    
    Returns:
        DataFrame with case_ID, method predictions, groundtruth (if provided), and rankings
    """
    # Start with case IDs from all methods
    all_cases = set()
    for df in results_by_method.values():
        all_cases.update(df['Case'].values)
    
    # Don't sort by Case here - we'll sort by ranking at the end
    comparison_df = pd.DataFrame({'Case': list(all_cases)})
    
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

    # Reorder columns: Case, groundtruth (if exists), methods (dG, Kd), rankings
    cols = ['Case']
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
    
    # Sort by ranking: groundtruth rank if available, otherwise use first method's rank
    if 'groundtruth_dG' in comparison_df.columns:
        # Calculate groundtruth rank (lower/more negative dG = better binding = rank 1)
        groundtruth_rank_col = 'groundtruth_rank'
        comparison_df[groundtruth_rank_col] = comparison_df['groundtruth_dG'].rank(ascending=True, method='min')
        comparison_df[groundtruth_rank_col] = comparison_df[groundtruth_rank_col].astype("Int64")
        # Sort by groundtruth rank, then by Case for ties
        comparison_df = comparison_df.sort_values(by=[groundtruth_rank_col, 'Case'], na_position='last')
        # Remove the temporary groundtruth_rank column (keep only method ranks)
        comparison_df = comparison_df.drop(columns=[groundtruth_rank_col])
    elif len(results_by_method) > 0:
        # Use first method's rank for sorting
        first_method = list(results_by_method.keys())[0]
        rank_col = f'{first_method}_rank'
        if rank_col in comparison_df.columns:
            # Sort by method rank, then by Case for ties
            comparison_df = comparison_df.sort_values(by=[rank_col, 'Case'], na_position='last')
    else:
        # No methods or groundtruth, just sort by Case
        comparison_df = comparison_df.sort_values(by=['Case'])
    
    # Reset index after sorting
    comparison_df = comparison_df.reset_index(drop=True)
    
    return comparison_df


@celery.task(bind=True, queue='affinity')
def affinity_single_pdb(self, pdb_file_id: str, metadata_dict: dict, 
                        method: str, temperature: float, task_id: str):
    """
    Process a single PDB file - called as subtask for parallel execution.
    
    This subtask is invoked by the main affinity() task using Celery groups
    to enable parallel processing across multiple worker pods.
    
    Args:
        pdb_file_id: File ID for the PDB file (from File.get_file())
        metadata_dict: Metadata dictionary for this case (must contain case_id as key)
        method: Method to use ('baseline', 'ensemble', 'variable_dielectric')
        temperature: Temperature in Kelvin
        task_id: Parent task ID for status updates
    
    Returns:
        Dict with 'Case', 'Predicted_dG', 'Predicted_Kd_nM', 'success' keys
    """
    try:
        # Get file object
        file_obj = File.get_file(pdb_file_id)
        if not file_obj or not file_obj.filepath:
            raise FileNotFoundError(f"File not found for file_id={pdb_file_id}")
        
        # Download file to temporary location
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdb_file_local = download_gcs_file_to_temp(file_obj.filepath, temp_path)
            
            # Extract case ID
            case_id = extract_case_id(str(pdb_file_local))
            
            # Get chain info from metadata
            # metadata_dict should be keyed by case_id
            if case_id not in metadata_dict:
                # Fallback: if metadata_dict is the full metadata, try direct access
                chain_info = metadata_dict.get(case_id)
                if chain_info is None:
                    raise ValueError(f"Case {case_id} not found in metadata")
            else:
                chain_info = metadata_dict[case_id]
            
            receptor_chains = chain_info["receptor_chains"]
            ligand_chains = chain_info["ligand_chains"]
            
            # Run MM/GBSA calculation
            result = run_mmgbsa(
                str(pdb_file_local),
                receptor_chains,
                ligand_chains,
                method=method,
                temperature=temperature,
                skip_fixing=False,
            )
            
            return {
                "Case": case_id,
                "Predicted_dG": result.get("dg_bind"),
                "Predicted_Kd_nM": result.get("kd_nm"),
                "success": True
            }
            
    except Exception as e:
        logger.error(f"Error in affinity_single_pdb for file_id={pdb_file_id}: {e}", exc_info=True)
        # Try to extract case_id for error reporting
        try:
            file_obj = File.get_file(pdb_file_id)
            if file_obj:
                # Try to extract from filename as fallback
                case_id = extract_case_id(file_obj.filename)
            else:
                case_id = "unknown"
        except:
            case_id = "unknown"
        
        return {
            "Case": case_id,
            "error": str(e),
            "success": False
        }


@celery.task(bind=True, queue='affinity')
def affinity_aggregate_method(self, subtask_results: List[Dict], method: str, task_id: str):
    """
    Callback task for Celery chord - aggregates results from parallel subtasks.
    This is called automatically by Celery when all subtasks in the group complete.
    
    Args:
        subtask_results: List of results from affinity_single_pdb subtasks (automatically passed by chord)
        method: Method name that was run
        task_id: Parent task ID for status updates
    
    Returns:
        Dict with aggregated results for this method
    """
    logger.info(f"affinity_aggregate_method called for method={method}, task_id={task_id}, results_count={len(subtask_results) if subtask_results else 0}")
    method_results = []
    method_errors = []
    
    for result in subtask_results:
        if result and result.get("success", False):
            method_results.append({
                "Case": result["Case"],
                "Predicted_dG": result["Predicted_dG"],
                "Predicted_Kd_nM": result["Predicted_Kd_nM"]
            })
        else:
            case_id = result.get("Case", "unknown") if result else "unknown"
            error_msg = result.get("error", "Unknown error") if result else "Task returned None"
            full_error_msg = f"Error processing {case_id} with {method}: {error_msg}"
            method_errors.append(full_error_msg)
            emit_status(task_id, f"error: {full_error_msg}")
    
    # Store results in Redis for main task to retrieve
    r = get_redis_connection()
    results_key = f"affinity:results:{task_id}:{method}"
    errors_key = f"affinity:errors:{task_id}:{method}"
    
    if method_results:
        # Convert to JSON-serializable format
        results_data = pd.DataFrame(method_results).to_dict('records')
        r.setex(results_key, 3600, json.dumps(results_data))  # 1 hour TTL
    
    if method_errors:
        r.setex(errors_key, 3600, json.dumps(method_errors))  # 1 hour TTL
    
    return {
        "method": method,
        "results_count": len(method_results),
        "errors_count": len(method_errors),
        "completed": True
    }


@celery.task(bind=True, queue='affinity')
def affinity(self, payload):
    """
    Calculate binding affinity for multiple PDB complex files using multiple methods.
    
    Input:
    - file_ids: List of GCS file references
      - Must include at least one PDB file (.pdb)
      - Must include exactly one metadata file (.json or .csv)
      - Optional: groundtruth CSV file (must have 'Case' and 'ΔG (kcal/mol)' or 'Experimental_dG' columns)
    - user_query: Optional user query string. Methods will be extracted from query if present.
      - Recognizes: "Baseline", "Ensemble", "Variable Dielectric" (or "VD")
      - Example: "use baseline and ensemble" or "run variable dielectric"
    - methods: Optional explicit list of methods to run (overrides user_query extraction).
      - Available: "baseline", "ensemble", "variable_dielectric"
      - Default: If neither user_query nor methods provided, runs all available methods
    - temperature: Optional temperature in Kelvin for MD simulation and Kd conversion.
      - Default: BODY_TEMPERATURE (310.15 K / 37°C) for human drug screening
      - Use ROOM_TEMPERATURE (298.15 K / 25°C) for laboratory comparisons
    
    Output:
    - CSV file with case_ID, predicted values by each method, optional groundtruth, and rankings
    """
    try:
        # Record start time for execution time tracking
        start_time = time.time()
        
        # --- Platform connections ---
        r = get_redis_connection()
        task_id = payload.get("task_id")
        user_id = payload.get("user_id")

        if not all([task_id, user_id]):
            raise ValueError(f"Missing required fields. task_id={task_id}, user_id={user_id}")

        # --- Inputs ---
        file_ids = payload.get("file_ids", [])
        user_query = payload.get("user_query", "")
        
        # Extract temperature from user query or use default
        if user_query:
            # Try to extract temperature from user query
            extracted_temp = extract_temperature_from_query(user_query)
            temperature = extracted_temp if extracted_temp is not None else BODY_TEMPERATURE
            if extracted_temp is not None:
                emit_status(task_id, f"extracted temperature from query: {temperature:.2f} K ({temperature - 273.15:.1f}°C)")
        else:
            # Default to body temperature for human drug screening
            temperature = BODY_TEMPERATURE
        
        # --- Determine methods to run ---
        # Default: all available methods
        all_available_methods = ["baseline", "ensemble", "variable_dielectric"]
        
        if user_query:
            # Extract methods from user query
            emit_status(task_id, "extracting method preferences from query")
            extracted_methods = extract_methods_from_query(user_query)
            if extracted_methods:
                methods_to_run = extracted_methods
                emit_status(task_id, f"using methods from query: {', '.join(methods_to_run)}")
            else:
                # No methods found in query, use all available
                methods_to_run = all_available_methods
                emit_status(task_id, "no methods specified in query, using all available methods")
        else:
            # No query and no explicit methods, use all available
            methods_to_run = ["ensemble"]

        if not file_ids:
            raise ValueError("No file_ids provided. Must include at least one PDB file and one metadata file.")

        emit_status(task_id, "parsing file inputs")

        # --- Get file metadata from database ---
        files = []
        for file_id in file_ids:
            file_obj = File.get_file(file_id)
            if not file_obj or not file_obj.filepath:
                raise FileNotFoundError(f"File not found for file_id={file_id}")
            files.append(file_obj)

        # --- Separate files by type ---
        pdb_files = [f for f in files if f.filename.lower().endswith('.pdb')]
        metadata_files = [f for f in files if 'metadata' in f.filename.lower()]
        groundtruth_files = [f for f in files if 'groundtruth' in f.filename.lower() or 'experimental' in f.filename.lower()]

        if len(pdb_files) == 0:
            raise ValueError("No PDB files found in file_ids. Must include at least one PDB file.")

        if len(metadata_files) != 1:
            raise ValueError(
                f"Exactly one metadata file required (JSON or CSV), found {len(metadata_files)}. "
                f"Found files: {[f.filename for f in metadata_files]}"
            )

        metadata_file = metadata_files[0]
        groundtruth_file = groundtruth_files[0] if groundtruth_files else None

        emit_status(task_id, f"processing {len(pdb_files)} PDB file(s) with methods: {', '.join(methods_to_run)}")

        # --- Create temporary directory for processing ---
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # --- Download files from GCS to temp directory ---
            emit_status(task_id, "downloading files from GCS")
            metadata_local = download_gcs_file_to_temp(metadata_file.filepath, temp_path)
            pdb_files_local = [
                download_gcs_file_to_temp(f.filepath, temp_path)
                for f in pdb_files
            ]
            
            groundtruth_local = None
            if groundtruth_file:
                groundtruth_local = download_gcs_file_to_temp(groundtruth_file.filepath, temp_path)

            # --- Load and parse metadata ---
            emit_status(task_id, "loading metadata")
            metadata = load_metadata(str(metadata_local))

            # --- Validate all PDB files have metadata entries ---
            validate_metadata([str(f) for f in pdb_files_local], metadata)

            # --- Load groundtruth if provided ---
            groundtruth_df = None
            if groundtruth_local:
                emit_status(task_id, "loading groundtruth")
                # Try comma-separated first (most common), then tab-separated
                try:
                    gt_df = pd.read_csv(groundtruth_local, sep=',')
                    # Check if we got a single column (likely wrong separator)
                    if len(gt_df.columns) == 1 and ',' in str(gt_df.iloc[0, 0] if len(gt_df) > 0 else ''):
                        gt_df = pd.read_csv(groundtruth_local, sep='\t')
                except:
                    try:
                        gt_df = pd.read_csv(groundtruth_local, sep='\t')
                    except:
                        # Last resort: let pandas auto-detect
                        gt_df = pd.read_csv(groundtruth_local)
                
                # Handle case ID column (accept both 'Case' and 'case_ID')
                case_col = None
                if 'Case' in gt_df.columns:
                    case_col = 'Case'
                elif 'case_ID' in gt_df.columns:
                    case_col = 'case_ID'
                    gt_df = gt_df.rename(columns={'case_ID': 'Case'})
                else:
                    raise ValueError("Groundtruth file must have 'Case' or 'case_ID' column")
                
                # Handle experimental dG column
                exp_dg_col = None
                # First, check for explicit column names
                if 'ΔG (kcal/mol)' in gt_df.columns:
                    exp_dg_col = 'ΔG (kcal/mol)'
                elif 'Experimental_dG' in gt_df.columns:
                    exp_dg_col = 'Experimental_dG'
                else:
                    # Auto-detect: find first numeric column (excluding the case column)
                    numeric_cols = gt_df.select_dtypes(include=[np.number]).columns.tolist()
                    # Remove case column if it's numeric (shouldn't be, but just in case)
                    if 'Case' in numeric_cols:
                        numeric_cols.remove('Case')
                    
                    if len(numeric_cols) > 0:
                        exp_dg_col = numeric_cols[0]
                        emit_status(task_id, f"auto-detected experimental dG column: {exp_dg_col}")
                    else:
                        raise ValueError(
                            "Groundtruth file must have a numeric column for experimental dG values. "
                            "Expected column names: 'ΔG (kcal/mol)', 'Experimental_dG', or any numeric column."
                        )
                
                # Rename to standard column name
                if exp_dg_col != 'Experimental_dG':
                    gt_df = gt_df.rename(columns={exp_dg_col: 'Experimental_dG'})
                
                # Ensure Case column exists (already renamed if needed)
                groundtruth_df = gt_df[['Case', 'Experimental_dG']].copy()

            # --- Run each method ---
            # Note: We don't pre-clean PDBs to preserve chain IDs.
            # MM/GBSA functions will handle cleaning internally (skip_fixing=False)
            # Use Celery groups for parallel execution across multiple worker pods
            results_by_method = {}
            errors_by_method = {}

            # Store results in a way that can be accessed by callback
            # We'll use Redis or pass via task state
            
            # Use Celery chord - the proper way to aggregate parallel task results
            # Chord = group + callback that automatically receives all results
            chord_callbacks = {}
            
            for method in methods_to_run:
                emit_status(task_id, f"running {method} method")
                emit_status(task_id, f"dispatching {len(pdb_files)} subtasks for {method} method")
                
                try:
                    # Create chord: group of subtasks + callback
                    # IMPORTANT: Convert generator to list (required for chords to work reliably)
                    # The callback receives all results as first argument automatically
                    header_tasks = [
                        affinity_single_pdb.s(
                            pdb_file.file_id,
                            metadata,
                            method,
                            temperature,
                            task_id
                        )
                        for pdb_file in pdb_files
                    ]
                    
                    # Create callback signature - results will be prepended automatically by Celery
                    chord_callback = affinity_aggregate_method.s(method, task_id)
                    
                    # Create chord with list (not generator) for reliability
                    chord_task = chord(group(header_tasks), chord_callback)
                    
                    # Execute chord - callback will be triggered automatically
                    chord_result = chord_task.apply_async()
                    chord_callbacks[method] = chord_result
                    logger.info(f"Chord created for method={method}, callback_task_id={chord_result.id}")
                    
                except Exception as e:
                    error_msg = f"Error dispatching {method} method: {str(e)}"
                    errors_by_method[method] = [error_msg]
                    emit_status(task_id, f"error: {error_msg}")
                    logger.error(f"Error in affinity task for method {method}: {error_msg}", exc_info=True)
            
            # Wait for all chord callbacks to complete
            emit_status(task_id, "waiting for all methods to complete")
            timeout_seconds = len(methods_to_run) * len(pdb_files) * 30 * 60  # 30 min per file per method
            wait_start_time = time.time()
            last_log_time = wait_start_time
            
            while time.time() - wait_start_time < timeout_seconds:
                all_complete = True
                for method in methods_to_run:
                    if method not in chord_callbacks:
                        continue
                    callback_ready = chord_callbacks[method].ready()
                    if not callback_ready:
                        all_complete = False
                        # Log progress every 30 seconds
                        if time.time() - last_log_time > 30:
                            logger.info(f"Waiting for {method} callback to complete (task_id={task_id})")
                            last_log_time = time.time()
                        break
                
                if all_complete:
                    logger.info(f"All chord callbacks completed for task {task_id}")
                    break
                time.sleep(2)  # Check every 2 seconds
            else:
                # Timeout reached
                logger.warning(f"Timeout waiting for chord callbacks for task {task_id}")
                emit_status(task_id, "warning: timeout waiting for some methods to complete")
            
            # Retrieve results from Redis (stored by callback tasks)
            for method in methods_to_run:
                if method not in chord_callbacks:
                    continue
                
                try:
                    results_key = f"affinity:results:{task_id}:{method}"
                    errors_key = f"affinity:errors:{task_id}:{method}"
                    
                    # Get results from Redis (stored by callback)
                    results_data = r.get(results_key)
                    errors_data = r.get(errors_key)
                    
                    if results_data:
                        method_results = json.loads(results_data)
                        results_by_method[method] = pd.DataFrame(method_results)
                    
                    if errors_data:
                        method_errors = json.loads(errors_data)
                        errors_by_method[method] = method_errors
                    else:
                        errors_by_method[method] = []
                        
                except Exception as e:
                    error_msg = f"Error retrieving results for {method}: {str(e)}"
                    errors_by_method[method] = [error_msg]
                    logger.error(f"Error retrieving results for {method}: {error_msg}", exc_info=True)

            # --- Check if we have any results ---
            if not results_by_method:
                raise ValueError("No methods completed successfully. Check errors.")

            # --- Generate comparison table ---
            emit_status(task_id, "generating comparison table")
            logger.info(f"[DEBUG] Generating comparison table for task {task_id}")
            logger.info(f"[DEBUG] results_by_method keys: {list(results_by_method.keys())}")
            logger.info(f"[DEBUG] results_by_method shapes: {[(k, v.shape) for k, v in results_by_method.items()]}")
            logger.info(f"[DEBUG] groundtruth_df is {'present' if groundtruth_df is not None else 'None'}")
            if groundtruth_df is not None:
                logger.info(f"[DEBUG] groundtruth_df shape: {groundtruth_df.shape}, columns: {list(groundtruth_df.columns)}")
            try:
                comparison_df = generate_comparison_table(results_by_method, groundtruth_df)
                logger.info(f"[DEBUG] Comparison table generated successfully, shape: {comparison_df.shape}")
                logger.info(f"[DEBUG] Comparison table columns: {list(comparison_df.columns)}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to generate comparison table: {e}", exc_info=True)
                raise

            # --- Create output CSV ---
            output_csv = temp_path / "affinity_comparison.csv"
            comparison_df.to_csv(output_csv, index=False)

            # --- Generate and emit summary message ---
            emit_status(task_id, "sending message")
            logger.info(f"[DEBUG] Starting message generation for task {task_id}")
            logger.info(f"[DEBUG] comparison_df shape: {comparison_df.shape}, columns: {list(comparison_df.columns)}")
            logger.info(f"[DEBUG] methods_to_run: {methods_to_run}, num_pdb_files: {len(pdb_files)}, temperature: {temperature}")
            
            try:
                # Calculate execution time
                execution_time = time.time() - start_time
                logger.info(f"[DEBUG] Task execution time: {execution_time:.2f} seconds")
                
                logger.info(f"[DEBUG] Calling generate_affinity_summary...")
                summary_message = generate_affinity_summary(comparison_df, methods_to_run, len(pdb_files), temperature, execution_time)
                logger.info(f"[DEBUG] Summary message generated successfully, length: {len(summary_message)} characters")
                logger.info(f"[DEBUG] Summary message preview (first 500 chars): {summary_message[:500]}")
                
                logger.info(f"[DEBUG] Creating MessageSchema...")
                message = MessageSchema(role="assistant", content=summary_message)
                logger.info(f"[DEBUG] MessageSchema created successfully")
                
                logger.info(f"[DEBUG] Converting message to dict...")
                message_dict = message.model_dump()
                logger.info(f"[DEBUG] Message dict created, keys: {list(message_dict.keys())}")
                
                logger.info(f"[DEBUG] Calling emit_task_message for task {task_id}...")
                emit_task_message(task_id, message_dict)
                logger.info(f"[DEBUG] emit_task_message call completed without exception")
                logger.info(f"Successfully emitted summary message for task {task_id} (message length: {len(summary_message)} chars)")
                emit_status(task_id, "message sent successfully")
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                logger.error(f"[ERROR] Message generation/emission failed for task {task_id}")
                logger.error(f"[ERROR] Error type: {error_type}")
                logger.error(f"[ERROR] Error message: {error_msg}")
                logger.error(f"[ERROR] Full traceback:", exc_info=True)
                # Emit a status update so user knows something went wrong
                emit_status(task_id, f"error: failed to generate/send summary message: {error_type}: {error_msg}")
                # Re-raise the exception so the task fails - message generation is critical
                logger.error(f"[ERROR] Re-raising exception - task will fail because message generation is required")
                raise

            # --- Upload to GCS ---
            emit_status(task_id, "uploading results to GCS")
            gcs_storage = GCSFileStorage()
            gcs_path = f"tasks/{task_id}/affinity_comparison.csv"
            gcs_storage.upload_file(str(output_csv), gcs_path, content_type='text/csv')

            # --- Emit file event ---
            file_data = {
                "user_id": user_id,
                "filename": "affinity_comparison.csv",
                "filepath": gcs_path,
                "file_type": "csv",
                "content_type": "text/csv"
            }
            emit_task_file(task_id, file_data)

            # --- Report errors if any ---
            all_errors = []
            for method, method_errors in errors_by_method.items():
                if method_errors:
                    all_errors.extend([f"{method}: {e}" for e in method_errors])
            if all_errors:
                emit_status(task_id, f"completed with {len(all_errors)} error(s). Check results.")

        # --- Completion ---
        # Mark task as finished only after all critical steps complete:
        # 1. Comparison table generated ✓
        # 2. Summary message generated and emitted ✓
        # 3. Results uploaded to GCS ✓
        logger.info(f"[DEBUG] All steps completed successfully for task {task_id}, marking as finished")
        finished_at = Task.mark_finished(task_id)
        logger.info(f"[DEBUG] Task {task_id} marked as finished at {finished_at}")
        emit_status(task_id, "done")
        return {
            "done": True,
            "finished_at": finished_at,
            "methods_completed": list(results_by_method.keys()),
            "errors": len(all_errors) if 'all_errors' in locals() else 0
        }

    except Exception as e:
        # --- Error handling ---
        error_msg = str(e)
        logger.error(f"Error in affinity task: {error_msg}", exc_info=True)
        emit_status(task_id, f"error: {error_msg}", error_message=error_msg)
        raise  # Re-raise the exception so Celery knows the task failed

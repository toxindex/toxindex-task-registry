# Binding Affinity Calculation Task

Automated Celery task for calculating binding affinity of protein-ligand complexes using MM/GBSA (Molecular Mechanics/Generalized Born Surface Area) methods.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Input Requirements](#input-requirements)
5. [Methods](#methods)
6. [Output Format](#output-format)
7. [Deployment](#deployment)
8. [Usage](#usage)
9. [Configuration](#configuration)
10. [Performance](#performance)
11. [Development](#development)

---

## Overview

This task calculates binding affinity (ΔG) for protein-ligand complexes using MM/GBSA calculations. It processes multiple PDB complex files in parallel using Celery workers and supports multiple calculation methods for comparison.

**Key Capabilities:**
- Process multiple PDB complex files in parallel
- Support for three MM/GBSA methods: baseline, ensemble, variable_dielectric
- Automatic temperature extraction from user queries (body temp vs room temp)
- Method selection via user query or explicit parameters
- Ground truth comparison with ranking quality metrics
- GPU-accelerated calculations (OpenMM with CUDA)

---

## Features

- **Parallel Processing**: Celery-based distributed task execution across multiple worker pods
- **Multiple Methods**: Baseline, Ensemble, and Variable Dielectric MM/GBSA calculations
- **Temperature Control**: Automatic extraction from user queries or explicit specification (body temp: 310.15K / room temp: 298.15K)
- **Method Comparison**: Compare predictions across multiple methods with rankings
- **Ground Truth Validation**: Optional comparison with experimental data (Spearman correlation, Kendall's tau, Top-10 accuracy)
- **Kd Conversion**: Automatic conversion from ΔG (kcal/mol) to Kd (nM) at specified temperature
- **GPU Acceleration**: OpenMM with CUDA support for faster calculations
- **Comprehensive Reporting**: Summary messages with statistics and ranking quality metrics

---

## Architecture

### Task Structure

```
affinity/
├── affinity/
│   ├── affinity_celery.py      # Main Celery task implementation
│   ├── affinity_utils.py        # Utility functions (conversions, metrics)
│   ├── mmgbsa_utils.py         # MM/GBSA calculation functions
│   └── celery_worker_affinity.py # Celery worker setup
├── pyproject.toml              # Project dependencies
├── Dockerfile.affinity         # Docker image definition
├── deployment_affinity.yaml    # Kubernetes deployment
└── hpa_affinity.yaml           # Horizontal Pod Autoscaler config
```

### Task Flow

1. **Input Parsing**: Extract PDB files, metadata, and optional groundtruth from file_ids
2. **Method Selection**: Extract methods from user_query (using LLM) or use explicit methods parameter
3. **Temperature Extraction**: Extract temperature preference from user_query or use default (body temp)
4. **Parallel Execution**: Dispatch Celery subtasks for each PDB file × method combination
5. **Result Aggregation**: Collect results from all subtasks via Celery chords
6. **Comparison Table**: Generate comprehensive comparison table with rankings
7. **Summary Generation**: Create summary message with statistics and metrics
8. **Output**: Upload CSV results to GCS and emit file event

### Parallel Processing

The task uses Celery **chords** to enable parallel processing:
- **Group**: Multiple `affinity_single_pdb` subtasks run in parallel (one per PDB file)
- **Callback**: `affinity_aggregate_method` aggregates results when all subtasks complete
- **Multiple Methods**: Each method runs as a separate chord, enabling true parallelism

---

## Input Requirements

### Required Files

1. **PDB Files** (`.pdb`): One or more protein-ligand complex structures
   - Must contain both receptor and ligand chains
   - Case ID is extracted from filename (e.g., `1S78_complex.pdb` → `1S78`)

2. **Metadata File** (`.json` or `.csv`): Exactly one metadata file specifying chain assignments
   
   **JSON Format:**
   ```json
   {
     "1S78": {
       "receptor_chains": ["H", "L"],
       "ligand_chains": ["A"]
     },
     "2DD8": {
       "receptor_chains": ["H", "L"],
       "ligand_chains": ["S"]
     }
   }
   ```
   
   **CSV Format:**
   ```csv
   case_ID,receptor_chains,ligand_chains
   1S78,"H,L","A"
   2DD8,"H,L","S"
   ```

### Optional Files

3. **Ground Truth File** (`.csv`): Experimental binding affinity data for validation
   - Must contain `Case` (or `case_ID`) column
   - Must contain experimental ΔG values in column named:
     - `ΔG (kcal/mol)`, or
     - `Experimental_dG`, or
     - Any numeric column (auto-detected)

### Task Payload

```python
{
    "task_id": "string",           # Required: Task identifier
    "user_id": "string",           # Required: User identifier
    "file_ids": ["id1", "id2"],    # Required: List of GCS file references
    "user_query": "string",        # Optional: Natural language query
    "methods": ["baseline"],       # Optional: Explicit method list (overrides user_query)
    "temperature": 310.15          # Optional: Temperature in Kelvin (overrides user_query)
}
```

### User Query Examples

The task can extract method preferences and temperature from natural language queries:

**Method Selection:**
- "use baseline and ensemble" → `["baseline", "ensemble"]`
- "run variable dielectric" → `["variable_dielectric"]`
- "use VD" → `["variable_dielectric"]`
- "calculate binding affinity" → `["ensemble"]` (default)

**Temperature Selection:**
- "body temperature" or "37°C" → `310.15 K`
- "room temperature" or "25°C" → `298.15 K`
- "310 K" → `310.0 K`
- "37 C" → `310.15 K`

---

## Methods

### Baseline Method

- **Description**: Standard MM/GBSA calculation with default parameters
- **Speed**: Moderate (depends on system size)
- **Accuracy**: Good baseline for comparison
- **Use Case**: Standard binding affinity estimation

### Ensemble Method

- **Description**: MM/GBSA calculation with ensemble averaging
- **Speed**: Slower (multiple conformations)
- **Accuracy**: Higher (accounts for conformational flexibility)
- **Use Case**: Default method for most accurate results

### Variable Dielectric (VD) Method

- **Description**: MM/GBSA with variable dielectric constant
- **Speed**: Moderate
- **Accuracy**: Can improve accuracy for certain systems
- **Use Case**: Alternative method for comparison

**Note**: All methods use OpenMM for MD simulation and ANTIPASTI for GB/SA calculations.

---

## Output Format

### CSV Results File

The task generates `affinity_comparison.csv` with the following columns:

- `Case`: Case identifier (extracted from PDB filename)
- `groundtruth_dG`: Experimental ΔG (if ground truth provided)
- `{method}_dG`: Predicted ΔG in kcal/mol for each method
- `{method}_Kd_nM`: Predicted Kd in nM (calculated from ΔG at specified temperature)
- `{method}_rank`: Ranking by each method (1 = strongest binding, lower ΔG = better)

**Example:**
```csv
Case,groundtruth_dG,baseline_dG,baseline_Kd_nM,baseline_rank,ensemble_dG,ensemble_Kd_nM,ensemble_rank
1S78,-12.5,-11.8,1.234e-20,2,-12.1,2.456e-21,1
2DD8,-10.2,-9.5,3.789e-18,3,-9.8,4.567e-19,2
```

### Summary Message

The task emits a summary message with:
- Processing statistics (number of files, methods used, temperature)
- Execution time
- Method-specific statistics (average ΔG, range, Kd statistics)
- Ground truth comparison metrics (if provided):
  - Top-10 accuracy
  - Spearman rank correlation
  - Kendall's tau

---

## Deployment

### Docker Image

The task is containerized in `Dockerfile.affinity`:
- Base image: `us-docker.pkg.dev/toxindex/toxindex-backend/basegpu:latest`
- Includes: OpenMM, ANTIPASTI, CUDA support
- Entrypoint: Celery worker for `affinity` queue

### Kubernetes Deployment

**Deployment** (`deployment_affinity.yaml`):
- Replicas: 3 (for parallel processing)
- GPU: NVIDIA L4 (1 GPU per pod)
- Resources: 4 CPU, 8GB RAM, 1 GPU
- Platform: CUDA for OpenMM acceleration

**Horizontal Pod Autoscaler** (`hpa_affinity.yaml`):
- Scales based on Celery queue length
- Min replicas: 3
- Max replicas: 10

### Environment Variables

- `CELERY_BROKER_URL`: Redis broker URL
- `CELERY_RESULT_BACKEND`: Redis result backend URL
- `OPENMM_PLATFORM`: "CUDA" for GPU acceleration
- `OPENMM_PLUGIN_DIR`: Path to OpenMM plugins
- `LD_LIBRARY_PATH`: CUDA library paths

---

## Usage

### Via API/Task Registry

The task is invoked through the task registry system:

```python
# Example payload
payload = {
    "task_id": "task_123",
    "user_id": "user_456",
    "file_ids": [
        "pdb_file_1_id",
        "pdb_file_2_id",
        "metadata_file_id",
        "groundtruth_file_id"  # Optional
    ],
    "user_query": "use ensemble method at body temperature"
}
```

### Direct Celery Invocation

```python
from affinity.affinity_celery import affinity

result = affinity.delay(payload)
```

---

## Configuration

### Method Selection

1. **Via User Query**: Methods extracted using LLM (Gemini 2.5 Flash)
2. **Explicit Parameter**: `methods` list in payload
3. **Default**: `["ensemble"]` if no methods specified

### Temperature Selection

1. **Via User Query**: Temperature extracted using LLM or keyword matching
2. **Explicit Parameter**: `temperature` in Kelvin
3. **Default**: `BODY_TEMPERATURE` (310.15 K / 37°C)

### MM/GBSA Parameters

MM/GBSA calculations use ANTIPASTI with OpenMM:
- **Force Field**: AMBER14SB (proteins) + GAFF2 (ligands)
- **GB Model**: Generalized Born with surface area
- **Temperature**: Specified in task (default: 310.15 K)
- **PDB Fixing**: Automatic (missing atoms, hydrogens, etc.)

---

## Performance

### Typical Performance

- **Per PDB File**: 5-30 minutes (depends on system size and method)
- **Parallel Processing**: Multiple files processed simultaneously across worker pods
- **GPU Acceleration**: 5-10x speedup with CUDA vs CPU

### Optimization Tips

1. **Use GPU**: Ensure `OPENMM_PLATFORM=CUDA` and GPU nodes available
2. **Parallel Workers**: Scale HPA to match workload
3. **Method Selection**: Use `ensemble` for accuracy, `baseline` for speed
4. **Batch Processing**: Process multiple files in single task (parallelized automatically)

### Resource Requirements

- **CPU**: 4 cores per worker pod
- **Memory**: 8GB per worker pod
- **GPU**: 1 NVIDIA L4 per worker pod (recommended)
- **Storage**: Temporary storage for PDB files (cleaned after processing)

---

## Development

### Local Testing

```bash
# Install dependencies
cd tasks/affinity
uv pip install -e .

# Test MM/GBSA calculation
python -c "
from affinity.mmgbsa_utils import run_mmgbsa
result = run_mmgbsa(
    'test.pdb',
    receptor_chains=['H', 'L'],
    ligand_chains=['A'],
    method='baseline',
    temperature=310.15
)
print(result)
"
```

### Running Celery Worker Locally

```bash
# Set environment variables
export CELERY_BROKER_URL="redis://localhost:6379/0"
export CELERY_RESULT_BACKEND="redis://localhost:6379/0"
export OPENMM_PLATFORM="CUDA"  # or "CPU"

# Start worker
python -m celery -A affinity.celery_worker_affinity worker \
    --loglevel=info -Q affinity
```

### Dependencies

Key dependencies (see `pyproject.toml`):
- `openmm`: Molecular dynamics simulation
- `antipasti`: MM/GBSA calculations
- `pandas`: Data handling
- `numpy`: Numerical calculations
- `langchain-google-vertexai`: LLM for query extraction
- `torch`: Required for ANTIPASTI (with CUDA support)

---

## Notes

- **Case ID Extraction**: Automatically extracted from PDB filename (supports various naming conventions)
- **PDB Fixing**: Automatic fixing of missing atoms, hydrogens, and other issues
- **Kd Calculation**: Uses exponential relationship `Kd = exp(ΔG / (R × T)) × 10⁹ nM`
  - **Warning**: Kd is exponentially sensitive to ΔG errors (0.1 kcal/mol → ~1.2x error in Kd)
  - For ranking, consider using ΔG directly to avoid error amplification
- **Temperature Impact**: ~2-3x change in Kd per 10°C temperature difference
- **Ground Truth Comparison**: Requires at least 2 cases with experimental data for correlation metrics
- **Error Handling**: Failed PDB files are reported but don't stop the entire task

---

## License

See LICENSE file in project root.

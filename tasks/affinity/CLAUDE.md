# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ToxIndex Task Registry - Celery-based computational tasks for the ToxIndex platform. This is the **affinity task** subdirectory which calculates binding affinity (ΔG) for protein-ligand complexes using MM/GBSA methods with GPU acceleration.

## Commands

### Installation
```bash
cd tasks/affinity
uv pip install -e .
```

### Testing
```bash
# Full test suite (~3-4 hours)
python affinity/affinity_test.py

# Quick determinism test (~5-10 minutes)
python affinity/quick_determinism_test.py

# PDBFixer determinism test (~30 seconds)
python affinity/test_pdbfixer_determinism.py

# Ultra-quick shell test
bash affinity/ultra_quick_test.sh
```

### Running Celery Worker Locally
```bash
export CELERY_BROKER_URL="redis://localhost:6379/0"
export CELERY_RESULT_BACKEND="redis://localhost:6379/0"
export OPENMM_PLATFORM="CUDA"  # or "CPU"
python -m celery -A affinity.celery_worker_affinity worker --loglevel=info -Q affinity
```

### Docker Build and Deploy
```bash
# Build
docker build -f deployment/Dockerfile.affinity -t affinity:latest .

# Push to GCR
docker tag affinity:latest us-docker.pkg.dev/toxindex/toxindex-backend/affinity:latest
docker push us-docker.pkg.dev/toxindex/toxindex-backend/affinity:latest

# Deploy to Kubernetes
kubectl apply -f deployment/deployment_affinity.yaml
kubectl apply -f deployment/hpa_affinity.yaml
```

### Workflow Registration (from repo root)
```bash
cd sync && python seed_workflows.py
```

## Architecture

### Task Structure
- `affinity/affinity_celery.py` - Main Celery task entry point
- `affinity/mmgbsa_utils.py` - Core MM/GBSA calculations (OpenMM + ANTIPASTI)
- `affinity/affinity_utils.py` - Utility functions (ΔG→Kd conversion, metrics)
- `affinity/celery_worker_affinity.py` - Worker setup

### Parallel Processing Pattern
Uses Celery **chords** for parallelism:
1. `group` of `affinity_single_pdb` subtasks (one per PDB file)
2. `callback` via `affinity_aggregate_method` aggregates results
3. Multiple methods run as separate chords for true parallelism

### Platform Integration
Tasks communicate with the platform via:
- `emit_status(task_id, msg)` - Real-time status updates
- `emit_task_message(task_id, message_dict)` - Chat messages to user
- `emit_task_file(task_id, file_data)` - GCS file uploads
- `Task.mark_finished(task_id)` - Mark completion

### Determinism Strategy
PDBFixer introduces non-deterministic hydrogen placement causing 8-21 kcal/mol variation. Solution:
- Structure caching in `.mmgbsa_cache/` directory
- First run: Process with PDBFixer → save to cache
- Subsequent runs: Load from cache → deterministic output
- Additional: Coordinate normalization (6 decimals), CUDA deterministic settings, strict energy minimization (0.001 kJ/mol tolerance)

## Key Dependencies
- `openmm` - Molecular dynamics (install via conda for CUDA)
- `antipasti` - MM/GBSA energy calculations
- `pdbfixer` - PDB structure fixing (from GitHub)
- `langchain-google-vertexai` - LLM for query extraction (Gemini 2.5 Flash)
- `numpy>=1.24.0,<2.0.0` - NumPy 1.x required for compatibility

## Code Style
- Black: line-length=100, target-version=py312
- Ruff: line-length=100, target-version=py312

## Input/Output

### Required Inputs
1. PDB files (`.pdb`) - Protein-ligand complex structures
2. Metadata file (`.json` or `.csv`) - Chain assignments per case:
   ```json
   {"1S78": {"receptor_chains": ["H", "L"], "ligand_chains": ["A"]}}
   ```

### Output
- `affinity_comparison.csv` with columns: Case, groundtruth_dG, {method}_dG, {method}_Kd_nM, {method}_rank

## Task Payload Structure
```python
{
    "task_id": "string",      # Required
    "user_id": "string",      # Required
    "file_ids": ["..."],      # Required: GCS file references
    "user_query": "string",   # Optional: Natural language query
    "methods": ["baseline"],  # Optional: explicit method list
    "temperature": 310.15     # Optional: Kelvin (default: body temp)
}
```

## Available Methods
- `baseline` - Standard MM/GBSA
- `ensemble` - With ensemble averaging (default, more accurate)
- `variable_dielectric` - Variable dielectric constant

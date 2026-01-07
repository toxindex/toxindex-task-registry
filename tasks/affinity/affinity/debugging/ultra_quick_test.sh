#!/bin/bash
# Ultra-quick test: Run MMGBSA twice and compare results
# Should complete in ~5-10 minutes

cd /home/kyu_insilica_co/toxindex-task-registry/tasks/affinity

# Clear cache
rm -rf affinity/sample_input/.mmgbsa_cache

echo "========================================"
echo "ULTRA-QUICK DETERMINISM TEST"
echo "Running MM/GBSA twice on same structure"
echo "========================================"

export PYTHONPATH=/home/kyu_insilica_co/toxindex-task-registry/tasks/affinity:$PYTHONPATH

# Run 1
echo ""
echo "RUN 1..."
affinity_cuda_env/bin/python -c "
from affinity.mmgbsa_utils import run_mmgbsa, load_metadata
metadata = load_metadata('affinity/sample_input/metadata_var1.csv')
case_id = '1S78'
result = run_mmgbsa(
    'affinity/sample_input/1S78_complex.pdb',
    metadata[case_id]['receptor_chains'],
    metadata[case_id]['ligand_chains'],
    method='baseline',
    temperature=310.15,
    platform_name='CUDA'
)
print(f'\\nRUN 1 RESULT: dG = {result[\"dg_bind\"]:.6f} kcal/mol')
print(f'E_complex = {result[\"e_complex\"]:.6f} kcal/mol')
with open('/tmp/run1_result.txt', 'w') as f:
    f.write(f'{result[\"dg_bind\"]},{result[\"e_complex\"]}')
"

# Run 2
echo ""
echo "RUN 2..."
affinity_cuda_env/bin/python -c "
from affinity.mmgbsa_utils import run_mmgbsa, load_metadata
metadata = load_metadata('affinity/sample_input/metadata_var1.csv')
case_id = '1S78'
result = run_mmgbsa(
    'affinity/sample_input/1S78_complex.pdb',
    metadata[case_id]['receptor_chains'],
    metadata[case_id]['ligand_chains'],
    method='baseline',
    temperature=310.15,
    platform_name='CUDA'
)
print(f'\\nRUN 2 RESULT: dG = {result[\"dg_bind\"]:.6f} kcal/mol')
print(f'E_complex = {result[\"e_complex\"]:.6f} kcal/mol')
with open('/tmp/run2_result.txt', 'w') as f:
    f.write(f'{result[\"dg_bind\"]},{result[\"e_complex\"]}')
"

# Compare
echo ""
echo "========================================"
echo "COMPARISON"
echo "========================================"

affinity_cuda_env/bin/python -c "
dg1, e1 = open('/tmp/run1_result.txt').read().strip().split(',')
dg2, e2 = open('/tmp/run2_result.txt').read().strip().split(',')

dg1, e1 = float(dg1), float(e1)
dg2, e2 = float(dg2), float(e2)

diff_dg = abs(dg1 - dg2)
diff_e = abs(e1 - e2)

print(f'Run 1: ΔG = {dg1:.6f} kcal/mol, E_complex = {e1:.6f} kcal/mol')
print(f'Run 2: ΔG = {dg2:.6f} kcal/mol, E_complex = {e2:.6f} kcal/mol')
print(f'')
print(f'Difference in ΔG: {diff_dg:.6f} kcal/mol')
print(f'Difference in E_complex: {diff_e:.6f} kcal/mol')
print(f'')

if diff_dg < 0.01:
    print('✓ EXCELLENT: Deterministic (< 0.01 kcal/mol)')
    exit(0)
elif diff_dg < 0.1:
    print('✓ GOOD: Mostly deterministic (< 0.1 kcal/mol)')
    exit(0)
elif diff_dg < 0.5:
    print('~ MODERATE: Some variation (< 0.5 kcal/mol)')
    exit(1)
else:
    print('✗ POOR: Non-deterministic (> 0.5 kcal/mol)')
    exit(1)
"

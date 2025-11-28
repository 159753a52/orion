#!/bin/bash

# GPT Model Profiling Script
# This script runs both NSYS and NCU profiling for the GPT model

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/venv/bin/python"
PROFILE_SCRIPT="${SCRIPT_DIR}/profile_gpt.py"
RESULTS_DIR="${SCRIPT_DIR}/profiling_results"
BATCHSIZE=${1:-4}

# Load CUDA environment
source /lihongliang/fangzl/envs/use-cuda-12.4.sh

# Create results directory
mkdir -p "${RESULTS_DIR}"

echo "============================================"
echo "GPT Model Profiling"
echo "Batch size: ${BATCHSIZE}"
echo "Results directory: ${RESULTS_DIR}"
echo "============================================"

# Step 1: Test run without profiling
echo ""
echo "[Step 1] Test run without profiling..."
${VENV_PYTHON} ${PROFILE_SCRIPT} --batchsize ${BATCHSIZE} --profile none

# Step 2: NSYS profiling
echo ""
echo "[Step 2] Running NSYS profiling..."
# Allow profiling
sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid' 2>/dev/null || echo "Warning: Could not set perf_event_paranoid"

nsys profile -w true -t cuda,nvtx,osrt \
    -s none -o "${RESULTS_DIR}/output_nsys" \
    --cudabacktrace=true \
    --capture-range=cudaProfilerApi \
    --stop-on-range-end=true \
    -f true -x true \
    ${VENV_PYTHON} ${PROFILE_SCRIPT} --batchsize ${BATCHSIZE} --profile nsys

echo "NSYS profiling complete. Output: ${RESULTS_DIR}/output_nsys.nsys-rep"

# Convert NSYS to CSV
echo "Converting NSYS output to CSV..."
nsys stats --report gputrace --format csv,column \
    --output "${RESULTS_DIR}/output_nsys_gputrace" \
    "${RESULTS_DIR}/output_nsys.nsys-rep" || echo "Warning: NSYS stats conversion failed"

# Step 3: NCU profiling
echo ""
echo "[Step 3] Running NCU profiling..."

# NCU with report file
ncu --set detailed --nvtx --nvtx-include "start/" \
    -o "${RESULTS_DIR}/output_ncu" \
    ${VENV_PYTHON} ${PROFILE_SCRIPT} --batchsize ${BATCHSIZE} --profile ncu

echo "NCU profiling complete. Output: ${RESULTS_DIR}/output_ncu.ncu-rep"

# NCU CSV output
echo "Running NCU with CSV output..."
ncu --csv --set detailed --nvtx --nvtx-include "start/" \
    ${VENV_PYTHON} ${PROFILE_SCRIPT} --batchsize ${BATCHSIZE} --profile ncu \
    > "${RESULTS_DIR}/output_ncu.csv" 2>&1

echo ""
echo "============================================"
echo "Profiling Complete!"
echo "Results saved in: ${RESULTS_DIR}"
echo ""
echo "Files generated:"
ls -la "${RESULTS_DIR}/"
echo "============================================"

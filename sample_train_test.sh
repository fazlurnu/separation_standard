#!/bin/bash

# --- Wait for any existing sweep_sampling_full* scripts to finish ---
while pgrep -f "sweep_sampling_full" > /dev/null; do
    echo "Another sweep_sampling_full* process is still running... waiting 600s."
    sleep 600
done

# --- Step 1: Sobol sampling ---
echo "Starting test sampling: sobol"
nohup python sweep_sampling_full_sobol.py \
    --script sample_ipr.py \
    --method sobol \
    --n-samples 256 \
    --out-root test_samples > sobol256.log 2>&1 &

# --- Wait for sobol run to finish before adaptive ---
echo "Waiting for sweep_sampling_full_sobol.py to finish..."
while pgrep -f "sweep_sampling_full_sobol.py" > /dev/null; do
    echo "Sobol sweep still running... waiting 600s."
    sleep 600
done

# --- Step 2: Adaptive sampling for training ---
echo "Starting adaptive sampling for training"
nohup python sweep_sampling_full_adaptive.py \
    --script sample_ipr.py \
    --method adaptive \
    --out-root train_samples \
    --n-init 32 \
    --batch-size 4 \
    --iterations 16 \
    --sobol-pool-size 1024 \
    --plot-grid-res 200 > adaptive.log 2>&1 &

echo "All sweeps launched successfully."
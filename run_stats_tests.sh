#!/bin/bash
# Script to test vLLM stats logging functionality

echo "=== Running vLLM Stats Logging Tests ==="
echo "This script verifies that stats logging is correctly configured."
echo 

# Set PYTHONPATH to include the vLLM directory
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run the test script
echo "Running test_vllm_stats_logging.py..."
python test_vllm_stats_logging.py

# Run the enhanced experiment script with stats enabled and disabled
echo
echo "=== Testing run_experiment_enhanced.py with stats enabled ==="
cd ../semantic-operators/ggr-experiment-pipeline
python -m src.experiment.run_experiment_enhanced sample_dataset.csv filter_bird_statistics \
    --model facebook/opt-125m \
    --batch-size 2 \
    --max-rows 4 \
    --max-model-len 512 \
    --gpu-memory 0.7 \
    --log-stats \
    --log-level INFO \
    --output-dir results_test

echo
echo "=== Testing run_experiment_enhanced.py with stats disabled ==="
python -m src.experiment.run_experiment_enhanced sample_dataset.csv filter_bird_statistics \
    --model facebook/opt-125m \
    --batch-size 2 \
    --max-rows 4 \
    --max-model-len 512 \
    --gpu-memory 0.7 \
    --disable-log-stats \
    --log-level INFO \
    --output-dir results_test

echo
echo "=== All tests completed ==="
echo "Check the output above for any errors or issues."

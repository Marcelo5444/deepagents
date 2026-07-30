#!/bin/bash
# nvalchemi benchmark runner with API keys from environment
# Usage: ./run_nvalchemi_benchmark.sh [model] [arm] [task]

set -euo pipefail

# API keys should be set in environment before running
# export NVIDIA_API_KEY="***"
# export LANGSMITH_API_KEY="***"
# export LANGSMITH_TRACING="true"

# Default values
MODEL="${1:-nvidia/nvidia/nemotron-3-super-v3}"
ARM="${2:-both}"
TASK="${3:-}"

cd /home/marcelo/deepagents-fork/libs/evals/nvalchemi_benchmarks

echo "=== nvalchemi Benchmark Runner ==="
echo "Model: $MODEL"
echo "Arm: $ARM"
echo "Task: ${TASK:-all}"
echo "=================================="

# Build command
CMD="python run_nvalchemi_benchmark.py --model $MODEL --arm $ARM"
if [ -n "$TASK" ]; then
    CMD="$CMD --task $TASK"
fi

echo "Running: $CMD"
echo ""

# Run the benchmark with longer timeout
timeout 1800 $CMD

echo ""
echo "=== Benchmark Complete ==="

# Show results summary
echo ""
echo "=== Verification of completed tasks ==="
for result in nvalchemi_benchmarks/work/*/with/result.json; do
    if [ -f "$result" ]; then
        task_dir=$(dirname "$result")
        task_name=$(basename "$task_dir")
        echo "Task: $task_name"
        cat "$result" | jq -r 'to_entries[] | "  \(.key): \(.value)"'
    fi
done
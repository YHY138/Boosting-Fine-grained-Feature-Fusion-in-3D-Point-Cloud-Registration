#!/usr/bin/env bash
# Run 3DMatch and 3DLoMatch tests and save logs
# Usage: ./rr_loop.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=python
# Prefer active conda/venv python if available
if [ -n "${CONDA_PREFIX-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
	PY="${CONDA_PREFIX}/bin/python"
elif [ -n "${VIRTUAL_ENV-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
	PY="${VIRTUAL_ENV}/bin/python"
fi
TEST_SCRIPT="${ROOT_DIR}/test.py"
MODEL="/path/to/your/model.pth"
LOG_DIR="${ROOT_DIR}/evaluate_logs"

mkdir -p "$LOG_DIR"

# Benchmarks to run (change as needed)
BENCHMARK="3DMatch"
# WEIGHTS=(0.8 0.7 0.6 0.5)
WEIGHTS=(0.0) # used for filter threshold ablation study

run_and_log() {
	local benchmark="$1"
    local weight_th="$2"
	local ts
	ts=$(date +%Y%m%d_%H%M%S)
	local logfile="${LOG_DIR}/test_${benchmark}_${ts}_ours_th${weight_th}.log"
	local stdout_file="${LOG_DIR}/test_${benchmark}_${ts}_ours_th${weight_th}.out"
	local stderr_file="${LOG_DIR}/test_${benchmark}_${ts}_ours_th${weight_th}.err"
	echo "========================" | tee -a "$logfile"
	echo "Running benchmark: $benchmark" | tee -a "$logfile"
	echo "Command: $PY -u $TEST_SCRIPT --dev --resume $MODEL --benchmark $benchmark --weight_th $weight_th" | tee -a "$logfile"

	# Ensure logfile exists
	touch "$logfile"

	# Run python unbuffered, stream stdout/stderr to .out/.err while
	# appending both streams to the combined logfile in real time.
	("$PY" -u "$TEST_SCRIPT" --dev --resume "$MODEL" --benchmark "$benchmark" --weight_th "$weight_th") \
		> >(tee -a "$stdout_file" >>"$logfile") \
		2> >(tee -a "$stderr_file" >>"$logfile" >&2)
	local rc=$?
	if [ $rc -ne 0 ]; then
		echo "Command failed for $benchmark with exit code $rc" | tee -a "$logfile"
		return $rc
	fi
	echo "Finished $benchmark (exit $rc)" | tee -a "$logfile"
}

echo "Starting batch runs (nohup-friendly). Logs -> $LOG_DIR"
for w in "${WEIGHTS[@]}"; do
  run_and_log "$BENCHMARK" "$w" || exit $?
done
# Make the script executable with: chmod +x rr_loop.sh

#!/bin/bash

# SM90 (Hopper) variant of run_ncu_mega_moe.sh
# Drives `tests/bench_mega_moe_sm90.py` with NCU, profiling the
# `sm90_fp8_mega_moe_impl` kernel for a single batch size.

set -e

num_processes=8
output_dir=work_sm90
python_args=()
for ((arg_idx = 1; arg_idx <= $#; ++arg_idx)); do
    arg="${!arg_idx}"
    case "$arg" in
        --num-processes)
            python_args+=("$arg")
            if ((arg_idx < $#)); then
                ((arg_idx++))
                num_processes="${!arg_idx}"
                python_args+=("$num_processes")
            fi
            ;;
        -h|--help)
            echo "Usage: $0 [--num-processes N] [--output DIR] [python args...]"
            exit 0
            ;;
        --num-processes=*)
            num_processes="${arg#*=}"
            python_args+=("$arg")
            ;;
        -o|--output)
            if ((arg_idx < $#)); then
                ((arg_idx++))
                output_dir="${!arg_idx}"
            fi
            ;;
        --output=*)
            output_dir="${arg#*=}"
            ;;
        *)
            python_args+=("$arg")
            ;;
    esac
done

echo "Python Args: ${python_args[*]}"
echo "Num Processes: $num_processes"
echo "Output Dir: $output_dir"
mkdir -p "$output_dir"

export DG_JIT_WITH_LINEINFO=1

echo "Warm up JIT cache"
python tests/bench_mega_moe_sm90.py --ncu-profile-only "${python_args[@]}"

sleep 2

ncu_args=(
    --config-file off
    --force-overwrite
    --kernel-name sm90_fp8_mega_moe_impl
    --import-source yes
    --replay-mode application
    --section SpeedOfLight
    --section LaunchStats
    --section SchedulerStats
    --section WarpStateStats
    --section MemoryWorkloadAnalysis
    --section InstructionStats
    --launch-skip 0
    --launch-count 1
    --clock-control none
    --kill yes
    --app-replay-buffer memory
)

echo "Run Job"

for ((i = 0; i < num_processes; ++i)); do
    ncu ${ncu_args[@]} -o "${output_dir%/}/mega-moe-sm90.$i" \
        python tests/bench_mega_moe_sm90.py \
            --local-rank-idx=$i \
            --ncu-profile-only \
            "${python_args[@]}" &
done

echo "Waiting"
wait
echo "Done"

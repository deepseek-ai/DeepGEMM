#!/bin/bash

# SM90 (Hopper) NVFP4 MegaMoE NCU harness.
# Drives `tests/bench_nvfp4_mega_moe_sm90.py`, profiling the generated
# `sm90_nvfp4_mega_moe_*_impl` kernel symbol(s) for a single batch size.

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
kernel_name="${NCU_KERNEL_NAME:-regex:.*sm90_nvfp4_mega_moe_.*_impl.*}"
launch_count="${NCU_LAUNCH_COUNT:-1}"

echo "Warm up JIT cache"
python tests/bench_nvfp4_mega_moe_sm90.py --ncu-profile-only "${python_args[@]}"

sleep 2

ncu_args=(
    --config-file off
    --force-overwrite
    --target-processes all
    --kernel-name "$kernel_name"
    --import-source yes
    --replay-mode application
    --section SpeedOfLight
    --section LaunchStats
    --section SchedulerStats
    --section WarpStateStats
    --section MemoryWorkloadAnalysis
    --section InstructionStats
    --launch-skip 0
    --launch-count "$launch_count"
    --clock-control none
    --kill yes
    --app-replay-buffer memory
)

echo "Run Job"
echo "NCU Kernel Name: $kernel_name"
echo "NCU Launch Count: $launch_count"

for ((i = 0; i < num_processes; ++i)); do
    ncu ${ncu_args[@]} -o "${output_dir%/}/mega-moe-sm90-nvfp4.$i" \
        python tests/bench_nvfp4_mega_moe_sm90.py \
            --local-rank-idx=$i \
            --ncu-profile-only \
            "${python_args[@]}" &
done

echo "Waiting"
wait
echo "Done"

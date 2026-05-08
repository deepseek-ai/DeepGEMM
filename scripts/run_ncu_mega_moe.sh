#!/bin/bash

set -e

# parse num-processes, output_dir and separate python args
num_processes=8
output_dir=work
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
mkdir -p $output_dir

export DG_JIT_WITH_LINEINFO=1 # for source counters

echo "Warm up JIT cache"
python tests/test_mega_moe.py --ncu-profile-only "${python_args[@]}"

sleep 2

ncu_args=(
    --config-file off
    --force-overwrite
    --kernel-name sm100_fp8_fp4_mega_moe_impl
    --import-source yes
    --replay-mode application
    --section PmSampling
    --section SourceCounters
    --rule LocalMemoryUsage
    --launch-skip 0
    --launch-count 1
    --lockstep-kernel-launch
    --communicator tcp
    --clock-control none
    --pm-sampling-interval 1000
    --pm-sampling-max-passes 1
    --disable-pm-warp-sampling
    --communicator-tcp-num-peers "$num_processes"
    --kill yes
    --app-replay-buffer memory
)

echo "Run Job"

for ((i = 0; i < num_processes; ++i)); do
    ncu ${ncu_args[@]} -o "${output_dir%/}/mega-moe.$i" \
        python tests/test_mega_moe.py \
            --local-rank-idx=$i \
            --ncu-profile-only \
            "${python_args[@]}" &
done

echo "Waiting"
wait
echo "Done"

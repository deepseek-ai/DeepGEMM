"""Run the same final-accuracy matrix through the actual public forward entry."""
import hashlib
import json
from pathlib import Path
import sys

from tests import bench_sm120_single_token as benchmark

original_paths = benchmark.paths
original_launch = benchmark.launch


def entry_paths(rt):
    labels, configs = original_paths(rt)
    return labels + ['entry'], configs


def entry_launch(rt, x, counts, slots, weights, label, configs):
    if label == 'entry':
        return rt.forward(x, counts, slots, weights)
    return original_launch(rt, x, counts, slots, weights, label, configs)


if __name__ == '__main__':
    # Only the test adapter changes: native launches and production forward are
    # unmodified. Existing benchmark correctness checks cover the added label.
    benchmark.paths = entry_paths
    benchmark.launch = entry_launch
    benchmark.main()
    output = Path(sys.argv[sys.argv.index('--output')+1])
    report = json.loads(output.read_text())
    report['public_forward_entry_measured'] = True
    report['entry_adapter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.write_text(json.dumps(report, indent=2)+'\n')

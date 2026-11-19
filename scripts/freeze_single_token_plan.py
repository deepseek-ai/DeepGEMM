"""Freeze T1 acceptance from accurate-kernel calibration, before new seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics


def upper_ratio(candidate, baseline):
    assert len(candidate) == len(baseline)
    rng = random.Random(25043)
    ratios = []
    for _ in range(2000):
        indices = [rng.randrange(len(candidate)) for _ in candidate]
        ratios.append(statistics.median(candidate[j] for j in indices) /
                      statistics.median(baseline[j] for j in indices))
    return sorted(ratios)[int(.95*(len(ratios)-1))]


def freeze(data):
    assert data['complete'] and data['repaired_arithmetic']
    plans = {}
    for model in sorted({r['model'] for r in data['rows']}):
        rows = [r for r in data['rows'] if r['model'] == model]
        assert len(rows) >= 2 and len({r['seed'] for r in rows}) == len(rows)
        states = [table for r in rows for table in r['tables'].values()]
        split = [n for n in states[0] if n.startswith('split-')]
        public = [n for n in states[0] if n.startswith('public-')]
        ratio = lambda name, state: state[name]['median_us']/min(state[n]['median_us'] for n in split+public)
        robust = lambda names: min(names, key=lambda name: max(ratio(name, s) for s in states))
        best_split, best_public = robust(split), robust(public)
        checks = []
        for row in rows:
            for condition, state in row['tables'].items():
                best = min(split+public, key=lambda n: state[n]['median_us'])
                candidate = state['candidate']; baseline = state[best]
                point = candidate['median_us']/baseline['median_us']
                upper = upper_ratio(candidate['samples_us'], baseline['samples_us'])
                checks.append(dict(seed=row['seed'], condition=condition, baseline=best,
                                   ratio=point, bootstrap_one_sided_95_upper=upper,
                                   passed=point < .97 and upper < .97))
        accepted = all(c['passed'] for c in checks)
        plans[model] = dict(key=rows[0]['key'], selected='candidate' if accepted else best_split,
            fusion_accepted=accepted, controls=[best_split, best_public], checks=checks,
            config=rows[0]['configs']['candidate'])
    return dict(complete=True, plans=plans, source_hashes=data['source_hashes'],
        calibration_seeds=sorted({r['seed'] for r in data['rows']}),
        boundary=data['boundary'], repaired_arithmetic=True,
        acceptance='candidate beats fastest measured ordinary/public control in all calibration states by 3%, one-sided bootstrap upper < .97; new-seed acceptance still required')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); result = freeze(json.loads(a.input.read_text()))
    result['calibration_sha256'] = hashlib.sha256(a.input.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({m: (v['selected'], v['fusion_accepted']) for m, v in result['plans'].items()}))


if __name__ == '__main__': main()

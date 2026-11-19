"""Audit frozen choices against independent data; never add calibration rejects."""
import argparse
import hashlib
import json
from pathlib import Path

from freeze_single_token_plan import upper_ratio


def audit(plan, data):
    assert plan['complete'] and data['complete'] and data['repaired_arithmetic']
    assert set(plan['calibration_seeds']).isdisjoint(r['seed'] for r in data['rows'])
    assert plan['source_hashes'] == data['source_hashes'], 'Revision changed between calibration and holdout'
    result = json.loads(json.dumps(plan))
    for model, frozen in result['plans'].items():
        rows = [r for r in data['rows'] if r['model'] == model]
        assert len(rows) >= 2 and len({r['seed'] for r in rows}) == len(rows)
        checks = []
        for row in rows:
            assert row['key'] == frozen['key']
            for condition, state in row['tables'].items():
                controls = [n for n in state if n.startswith(('split-', 'public-'))]
                best = min(controls, key=lambda n: state[n]['median_us'])
                candidate, baseline = state['candidate'], state[best]
                ratio = candidate['median_us']/baseline['median_us']
                upper = upper_ratio(candidate['samples_us'], baseline['samples_us'])
                checks.append(dict(seed=row['seed'], condition=condition, baseline=best,
                                   ratio=ratio, upper=upper, passed=ratio < .97 and upper < .97))
        frozen['independent_checks'] = checks
        frozen['independent_accepted'] = frozen['fusion_accepted'] and all(c['passed'] for c in checks)
    result['acceptance'] += '; independent audit may remove but never add support'
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--holdout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(json.loads(a.plan.read_text()), json.loads(a.holdout.read_text()))
    result['plan_sha256'] = hashlib.sha256(a.plan.read_bytes()).hexdigest()
    result['holdout_sha256'] = hashlib.sha256(a.holdout.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print({n: v['independent_accepted'] for n, v in result['plans'].items()})

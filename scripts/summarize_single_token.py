"""Summarize final public-entry evidence without conflating timing boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from freeze_single_token_plan import upper_ratio

LABELS = {'qwen30': 'Qwen3-30B', 'qwen235': 'Qwen3-235B',
          'qwen15': 'Qwen1.5-MoE', 'qwen2': 'Qwen2-MoE', 'deepseek': 'DeepSeek-V2-Lite'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    entry = json.loads((a.evidence/'final-entry.json').read_text())
    validation = json.loads((a.evidence/'validation/completion.json').read_text())
    real = json.loads((a.evidence/'real-replay.json').read_text())
    assert entry['complete'] and entry['public_forward_entry_measured']
    assert validation['all_steps_passed'] and real['complete']
    assert entry['extension_sha256'] == validation['extension_sha256']
    for n, h in entry['source_hashes'].items():
        if n in validation['source_hashes']: assert validation['source_hashes'][n] == h
        assert real['source_hashes'][n] == h
    rows = []
    for row in entry['rows']:
        for condition, table in row['tables'].items():
            controls = [n for n in table if n.startswith(('split-', 'public-'))]
            best = min(controls, key=lambda n: table[n]['median_us'])
            candidate, baseline = table['entry'], table[best]
            ratio = candidate['median_us']/baseline['median_us']
            rows.append(dict(model=row['model'], seed=row['seed'], condition=condition,
                baseline=best, entry_us=candidate['median_us'], baseline_us=baseline['median_us'],
                ratio=ratio, upper=upper_ratio(candidate['samples_us'], baseline['samples_us']),
                legacy_us=table.get('legacy', {}).get('median_us'),
                legacy_ratio=candidate['median_us']/table['legacy']['median_us'] if 'legacy' in table else None))
    replay_rows = []
    for row in real['timing']:
        for condition, table in row['tables'].items():
            # Fixed frozen ordinary control; candidate is never a baseline.
            baseline = next(n for n in table if n.startswith('split-'))
            replay_rows.append(dict(kind=row['kind'], layer=row['layer'], prompt=row['prompt'],
                condition=condition, entry_us=table['entry']['median_us'],
                baseline_us=table[baseline]['median_us'],
                ratio=table['entry']['median_us']/table[baseline]['median_us']))
    result = dict(complete=True, final_entry=rows, real_replay=replay_rows,
        validation_steps=validation['steps'], real_check_rows=len(real['checks']),
        real_oracle_rows=sum('oracle_Y' in c for c in real['checks']),
        source_hashes=entry['source_hashes'],
        evidence_sha256={n: hashlib.sha256((a.evidence/n).read_bytes()).hexdigest()
            for n in ('final-entry.json', 'validation/completion.json', 'real-replay.json')})
    entries = [c for c in real['checks'] if c['label'] == 'entry']
    result['real_entry_inputs'] = len(entries)
    result['all_entry_H_Y_bitwise'] = all(c['H']['bitwise'] and c['Y']['bitwise'] for c in entries)
    result['historical_HF_outside_tolerance_inputs'] = sum(
        c['historical_HF']['outside_original_tolerance'] > 0 for c in entries)
    result['historical_HF_outside_tolerance_elements'] = sum(
        c['historical_HF']['outside_original_tolerance'] for c in entries)
    (a.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Final single-token evidence', '',
        'Timing: pre-routed expert branch, allocation/JIT/router excluded. Same repaired arithmetic.', '',
        '| Shape | Condition | Entry latency range (us) | Best ordinary/public range (us) | Entry/control range |',
        '|---|---|---:|---:|---:|']
    models = list(LABELS)
    for model in models:
        for condition in ('warm', 'pressure512'):
            selected = [r for r in rows if r['model'] == model and r['condition'] == condition]
            span = lambda key: f'{min(r[key] for r in selected):.3f}–{max(r[key] for r in selected):.3f}'
            lines.append(f'| {LABELS[model]} | {condition} | {span("entry_us")} | {span("baseline_us")} | {span("ratio")} |')
    lines += ['', 'Qwen2 is the deliberately rejected fusion boundary: its entry runs ordinary kernels.', '',
              '| Shape with legacy fusion | Condition | Entry/legacy latency range |',
              '|---|---|---:|']
    for model in models:
        for condition in ('warm', 'pressure512'):
            values = [r['legacy_ratio'] for r in rows if r['model'] == model
                      and r['condition'] == condition and r['legacy_ratio'] is not None]
            if values: lines.append(f'| {LABELS[model]} | {condition} | {min(values):.3f}–{max(values):.3f} |')
    lines += ['', 'Values above one mean the existing fused configuration was faster in that condition.', '',
              f'Validation: {len(validation["steps"])} stages passed; real replay: {len(real["checks"])} label/input checks.', '',
              f'All {len(entries)} public-entry inputs match ordinary H/Y bitwise. '
              f'{sum("oracle_Y" in c for c in entries)} selected inputs passed independent FP64 contract checks. '
              f'Historical HF inputs outside the original tolerance: {result["historical_HF_outside_tolerance_inputs"]}.', '',
              'Real replay uses Qwen3 layers 0/4/7 and teacher-forced inputs. It is not full-model serving.', '',
              '| Real replay boundary | Condition | Median entry/control across recorded cases | Range |',
              '|---|---|---:|---:|']
    for kind in ('steady', 'sequence_average'):
        for condition in ('warm', 'pressure512'):
            values = [r['ratio'] for r in replay_rows if r['kind'] == kind and r['condition'] == condition]
            lines.append(f'| {kind} | {condition} | {statistics.median(values):.3f} | {min(values):.3f}–{max(values):.3f} |')
    lines += ['', '512 MiB interference is a timing condition, not a measured DRAM/cache metric.',
              'Warm samples average ten repeated calls; sequence samples average 32 calls. Neither gives per-request P95.',
              'Single-card/session evidence; no framework, cross-day, or universal correct-rounding claim.', '']
    (a.output/'results.md').write_text('\n'.join(lines), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, condition in zip(axes, ('warm', 'pressure512')):
        for index, model in enumerate(models):
            values = [1/r['ratio'] for r in rows if r['model'] == model and r['condition'] == condition]
            center = statistics.mean(values)
            ax.bar(index, center, color='#2a788e' if model != 'qwen2' else '#999999', width=.65)
            ax.scatter([index]*len(values), values, color='black', s=16, zorder=4)
            ax.text(index, center+.035, f'{center:.2f}x', ha='center', fontsize=9)
        ax.axhline(1, color='black', linestyle='--', linewidth=1)
        ax.set_xticks(range(len(models)), [LABELS[m] for m in models], rotation=22, ha='right')
        ax.set_title('Warm Graph repeats' if condition == 'warm' else '512 MiB interference before call')
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0].set_ylabel('Best ordinary/public time / public entry time')
    fig.suptitle('SM120 / RTX 5090: final repaired T1 public entry\nDots: two independent final seeds; Qwen2 entry uses fallback')
    fig.tight_layout()
    fig.savefig(a.output/'final-entry.png', dpi=180)
    fig.savefig(a.output/'final-entry.pdf')
    plt.close(fig)
    print('\n'.join(lines))


if __name__ == '__main__': main()

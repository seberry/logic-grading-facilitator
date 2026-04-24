import json, pathlib

root = pathlib.Path('working/tfl_gold_regressions_latest')
out = []
for d in sorted(root.iterdir()):
    r = d / 'gold_regression_result.json'
    m = d / 'gold_regression_metrics.json'
    if r.exists() and m.exists():
        res = json.loads(r.read_text('utf-8'))
        met = json.loads(m.read_text('utf-8'))
        out.append('=== ' + d.name + ' ===')
        out.append('passed: ' + str(res.get('passed')))
        out.append('summary: ' + str(res.get('run_summary')))
        out.append('error: ' + str(res.get('error', '')))
        metrics = met.get('metrics', {})
        if metrics:
            for key in ['expected_line_count', 'actual_line_count', 'formula_correct_line_count',
                        'justification_correct_line_count', 'both_correct_line_count',
                        'missing_line_count', 'extra_line_count', 'structure_mismatch_count']:
                out.append(key + ': ' + str(metrics.get(key)))
            for lr in metrics.get('line_records', []):
                exp = lr.get('expected')
                act = lr.get('actual')
                flags = []
                if not lr.get('formula_matches', True): flags.append('FORMULA_MISS')
                if not lr.get('justification_matches', True): flags.append('JUST_MISS')
                if not lr.get('structure_matches', True): flags.append('STRUCT_MISS')
                if lr.get('status') == 'missing_actual_line': flags.append('MISSING')
                if lr.get('status') == 'extra_actual_line': flags.append('EXTRA')
                if not flags: flags.append('OK')
                exp_str = (exp['wffstr'] + ' | ' + exp['jstr']) if exp else '(none)'
                act_str = (act['wffstr'] + ' | ' + act['jstr']) if act else '(none)'
                out.append('  line ' + str(lr.get('order_index', '?')) + ': ' + ' '.join(flags))
                out.append('    exp: ' + exp_str)
                out.append('    act: ' + act_str)
        out.append('')

pathlib.Path('working/latest_summary.txt').write_text('\n'.join(out), 'utf-8')
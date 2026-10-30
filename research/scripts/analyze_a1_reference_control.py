"""CPU analysis of the preregistered A1 development control audit; no G1 claim."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

CONTROLLERS = ('learned', 'zero_residual')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(rows):
    complete = [r for r in rows if r['complete_episode']]
    failure_controls = {}
    for row in complete:
        if row['failure_reason'] not in [None, 0]:
            failure_controls.setdefault(str(row['failure_reason']), []).append(row['controls'])
    return dict(recorded=len(rows), complete=len(complete),
                regular=sum(r['regular_success'] is True for r in complete),
                strict=sum(r['strict_success'] is True for r in complete),
                incomplete=len(rows)-len(complete),
                failure_reasons=dict(Counter(str(r['failure_reason']) for r in complete)),
                first_failure_control_by_reason={reason:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),minimum=min(v),maximum=max(v)) for reason,v in failure_controls.items()},
                mean_episode_controls=float(np.mean([r['controls'] for r in complete])) if complete else None,
                median_episode_controls=float(np.median([r['controls'] for r in complete])) if complete else None)


def task_bootstrap(task_values, task_strata, seed, n):
    """Resample whole tasks within strata, preserving all seeds/repeats per task."""
    tasks = sorted(task_values)
    values = np.asarray([task_values[t] for t in tasks], dtype=float)
    rng = np.random.default_rng(seed)
    indices = []
    for stratum in sorted(set(task_strata[t] for t in tasks)):
        ids = np.asarray([i for i, t in enumerate(tasks) if task_strata[t] == stratum])
        indices.append(rng.choice(ids, size=(n, len(ids)), replace=True))
    draws = values[np.concatenate(indices, axis=1)].mean(axis=1)
    return dict(mean=values.mean(axis=0).tolist(),
                percentile95=np.quantile(draws, [.025, .975], axis=0).T.tolist(),
                task_clusters=len(tasks), resamples=n, seed=seed)


def analyze(config, registration, rows, summary, job, configuration_sha256):
    tasks = config['task_ids']; seeds = config['development_seeds']
    assert len(set(tasks)) == len(tasks) and len(set(seeds)) == len(seeds)
    assert config['controllers'] == list(CONTROLLERS) and config['repeats'] == 2
    expected = set(itertools.product(tasks, seeds, CONTROLLERS, [1, 2]))
    assert len(expected) == config['maximum_conditions']
    assert registration['configuration'] == config
    assert registration['configuration_sha256'] == configuration_sha256
    assert registration['planned_conditions'] == len(expected)
    assert registration['policy_sha256'] and registration['checkpoint_sha256']
    assert job['name'] == config['run'] and job['phase'] == config['budget_phase']
    assert job['limit_seconds'] == config['maximum_gpu_seconds']
    by_key = {}
    for row in rows:
        key = (row['task'], row['seed'], row['controller'], row['repeat'])
        assert key in expected and key not in by_key, ('Unexpected/duplicate condition', key)
        assert row['stratum'] == config['task_strata'][row['task']]
        assert row['start_phase'] == 0
        assert isinstance(row['complete_episode'], bool)
        if row['complete_episode']:
            assert isinstance(row['regular_success'], bool) and isinstance(row['strict_success'], bool)
            assert not row['strict_success'] or row['regular_success']
            assert row['controls'] > 0 and row['finite'] is True
        else:
            assert row['regular_success'] is None and row['strict_success'] is None
        assert len(row['error_trace_m_rad']) == len(row['endpoint_metrics']) == len(row['contact_trace']) == row['controls']
        if row['controls']:
            errors = np.asarray(row['error_trace_m_rad']); metrics = np.asarray(row['endpoint_metrics'])
            assert errors.shape == (row['controls'], 2) and metrics.shape == (row['controls'], 6)
            assert np.isfinite(errors).all() and np.isfinite(metrics).all()
        by_key[key] = row
    for task, seed in itertools.product(tasks, seeds):
        group = [r for key, r in by_key.items() if key[:2] == (task, seed)]
        assert len({r['initial_state_context_memory_sha256'] for r in group}) <= 1, ('Pairing mismatch', task, seed)

    report = dict(status='INCOMPLETE_INCONCLUSIVE', formal_A1_complete=False, G1_claim=False,
                  quality_admission=False, recorded=len(rows), expected=len(expected),
                  missing_conditions=[list(k) for k in sorted(expected-set(by_key))],
                  by_controller={c:counts([r for r in rows if r['controller']==c]) for c in CONTROLLERS},
                  by_repeat={c:{str(p):counts([r for r in rows if r['controller']==c and r['repeat']==p]) for p in [1,2]} for c in CONTROLLERS},
                  by_stratum={s:{c:counts([r for r in rows if r['stratum']==s and r['controller']==c]) for c in CONTROLLERS} for s in sorted(set(config['task_strata'].values()))},
                  by_task={t:{c:counts([r for r in rows if r['task']==t and r['controller']==c]) for c in CONTROLLERS} for t in tasks},
                  checkpoint_sha256=registration['checkpoint_sha256'], parent_controls=registration['parent_controls'],
                  configuration_sha256=configuration_sha256,
                  interpretation='Development diagnostic, not PPO quality admission or G1. Repeats are not independent task/seed samples. Both controllers failing does not prove physical impossibility.')
    terminal_ok = job['status'] == 'finished' and job.get('returncode') == 0
    if summary:
        assert summary['conditions'] == len(rows) and summary['planned_conditions'] == len(expected)
        assert summary['checkpoint_sha256'] == registration['checkpoint_sha256']
        for c in CONTROLLERS:
            assert all(summary['counts'][c][k] == report['by_controller'][c][k] for k in ['recorded','complete','regular','strict'])
    complete_ok = (set(by_key) == expected and all(r['complete_episode'] for r in rows)
                   and summary and summary['status'] == 'complete' and summary['policy_unchanged'] is True
                   and terminal_ok)
    report['evidence_gate'] = dict(terminal_supervisor_ok=terminal_ok, complete_evidence=bool(complete_ok))
    if not complete_ok:
        report['paired_analysis'] = None
        return report

    # Every task/seed uses a common prefix across all four trajectories. No padded tails.
    pairs = []; task_vectors = {}
    columns = ['common_prefix_position_delta_mm','common_prefix_orientation_delta_deg',
               'regular_success_rate_delta','strict_success_rate_delta',
               'learned_repeat_abs_position_difference_mm','learned_repeat_abs_orientation_difference_deg',
               'zero_repeat_abs_position_difference_mm','zero_repeat_abs_orientation_difference_deg']
    for task in tasks:
        vectors = []
        for seed in seeds:
            group = {(c,p):by_key[(task,seed,c,p)] for c,p in itertools.product(CONTROLLERS,[1,2])}
            n = min(r['controls'] for r in group.values()); assert n > 0
            errors = {k:np.mean(r['error_trace_m_rad'][:n],axis=0)*[1000,180/np.pi] for k,r in group.items()}
            means = {c:np.mean([errors[(c,p)] for p in [1,2]],axis=0) for c in CONTROLLERS}
            delta = means['learned'] - means['zero_residual']
            success_delta = [np.mean([group[('learned',p)][key] for p in [1,2]])-np.mean([group[('zero_residual',p)][key] for p in [1,2]]) for key in ['regular_success','strict_success']]
            repeat_delta = [np.abs(errors[(c,1)]-errors[(c,2)]) for c in CONTROLLERS]
            vector = np.concatenate([delta,success_delta,*repeat_delta]); vectors.append(vector)
            endpoint_means = {c:np.mean([np.mean(group[(c,p)]['endpoint_metrics'][:n],axis=0) for p in [1,2]],axis=0).tolist() for c in CONTROLLERS}
            pairs.append(dict(task=task,seed=seed,stratum=config['task_strata'][task],common_controls=n,
                              controller_common_prefix_error_mm_deg={c:v.tolist() for c,v in means.items()},
                              endpoint_common_prefix_mean=endpoint_means,values=vector.tolist()))
        task_vectors[task] = np.mean(vectors,axis=0).tolist()
    plan = config['analysis_plan']
    report.update(status='COMPLETE_DEVELOPMENT_DIAGNOSTIC', paired_analysis=dict(
        columns=columns, delta_direction='learned minus zero residual; negative error differences favor learned',
        weighting='Equal time weight within common prefix; equal repeat, seed and task weights. Stratified task cluster bootstrap.',
        endpoint_metric_columns=registration['endpoint_metric_columns'], endpoint_metric_scope=registration['endpoint_metric_scope'],
        pairs=pairs, task_vectors=task_vectors,
        overall=task_bootstrap(task_vectors,config['task_strata'],plan['bootstrap_seed'],plan['bootstrap_resamples']),
        strata={s:task_bootstrap({t:v for t,v in task_vectors.items() if config['task_strata'][t]==s},config['task_strata'],plan['bootstrap_seed'],plan['bootstrap_resamples']) for s in sorted(set(config['task_strata'].values()))},
        interpretation='Descriptive paired intervals; no independent-repeat inference, causal training-effect estimate, physical-impossibility proof, or final-test claim.'))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(); read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    config=read(a.config); registration=read(a.directory/'registration.json')
    rows=read(a.directory/'conditions.json')
    summary=read(a.directory/'summary.json') if (a.directory/'summary.json').exists() else None
    jobs=[j for j in read(a.ledger)['jobs'] if j['name']==config['run']]; assert len(jobs)==1
    report=analyze(config,registration,rows,summary,jobs[0],digest(a.config))
    report['source_sha256']={str(p):digest(p) for p in [a.config,a.ledger,a.directory/'registration.json',a.directory/'conditions.json',Path(__file__)]}
    if summary:report['source_sha256'][str(a.directory/'summary.json')]=digest(a.directory/'summary.json')
    with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ['status','recorded','expected','by_controller','evidence_gate']},ensure_ascii=False))


if __name__=='__main__':main()

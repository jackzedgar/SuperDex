"""Paired development-only regression monitor; no success-threshold changes."""
import statistics

def paired_errors(current, baseline):
    by_key={(r['task'],r['seed']):r for r in baseline['all_conditions']}
    old=[];new=[];lengths=[]
    for row in current['all_conditions']:
        b=by_key[(row['task'],row['seed'])]
        n=min(len(row['error_trace_m_rad']),len(b['error_trace_m_rad']))
        assert n>0
        lengths.append(n)
        old.append([sum(x[k] for x in b['error_trace_m_rad'][:n])/n for k in [0,1]])
        new.append([sum(x[k] for x in row['error_trace_m_rad'][:n])/n for k in [0,1]])
    assert len(new)==len(by_key)
    return dict(baseline_mean=[statistics.mean(x[k] for x in old) for k in [0,1]],
                current_mean=[statistics.mean(x[k] for x in new) for k in [0,1]],
                paired_conditions=len(new),common_controls=sum(lengths),
                weighting='Equal condition weight after time mean on each paired common prefix; position m, orientation rad.')

def regression_check(current,baseline):
    pair=paired_errors(current,baseline)
    mean_len=lambda d:statistics.mean(x['controls'] for x in d['all_conditions'])
    # Treat a never-failed trace as right-censored at its executed horizon.
    first=lambda d:statistics.median(x['first_strict_failure_s'] if x['first_strict_failure_s'] is not None else .02*x['controls'] for x in d['all_conditions'])
    bcount=round(baseline['regular_success_rate']*baseline['n'])
    count=round(current['regular_success_rate']*current['n'])
    tests=dict(regular_loses_at_least4=count<=bcount-4,
               mean_episode_le80percent=mean_len(current)<=.8*mean_len(baseline),
               median_first_failure_no_later=first(current)<=first(baseline),
               common_position_ge120percent=pair['current_mean'][0]>=1.2*pair['baseline_mean'][0],
               common_rotation_ge120percent=pair['current_mean'][1]>=1.2*pair['baseline_mean'][1])
    return dict(joint_degradation=all(tests.values()),tests=tests,paired_common_prefix=pair,
                baseline_regular=bcount,current_regular=count,
                baseline_mean_episode_controls=mean_len(baseline),current_mean_episode_controls=mean_len(current))

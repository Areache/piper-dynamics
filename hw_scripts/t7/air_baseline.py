"""Offline, same-path empty-air current calibration. Never connects to hardware.

build -> candidate; independent --air-validate run -> qualify -> usable baseline.
No extrapolation, online adaptation, speed changes, or reuse across setup contexts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ENDPOINT_POLICY = 'final_command_joint_0p1deg_index_0p5mm_v1'
ENDPOINT_LAG_MM = 0.5
ENDPOINT_JOINT_DEG = 0.1


def baseline_endpoint_reached(k_cmd, k_meas, k_max, q, Q, step_mm=0.25,
                              tolerance_mm=ENDPOINT_LAG_MM):
    """Bound joint tracking error AND nearest-waypoint lag after the final command.

    Waypoint lag is a joint-space index, not a Cartesian measurement. A small
    servo bias can move that index by two 0.25 mm nodes even with every joint
    within 0.1 degree. This never permits commands beyond calibrated coverage.
    """
    q, target = np.asarray(q), np.asarray(Q[k_max])
    return bool(k_cmd >= k_max and np.isfinite(step_mm) and step_mm > 0
                and 0 <= (k_max-k_meas)*step_mm <= tolerance_mm + 1e-9
                and q.shape == target.shape == (6,) and np.isfinite(q).all()
                and np.max(np.abs(q-target)) <= np.radians(ENDPOINT_JOINT_DEG))


class BaselineError(ValueError):
    pass


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(plan, speed, speed_pct, context):
    if not isinstance(context, str) or not context.strip():
        raise BaselineError('a nonempty --baseline-context is required for this powered session/setup')
    return {'plan_sha256': digest(plan), 'arm': plan['arm'], 'motion': 'stream',
            'speed_mm_s': float(speed), 'speed_pct': int(speed_pct), 'context': context,
            'detect_joints': [2, 3], 'avg_window_s': 0.2, 'floor_N': 2.0,
            'lp_n': int(plan['detect']['lp_n'])}


def endpoint(plan, profile=None):
    planned = min(len(plan['q']) - 1, int(plan['max_depth_mm'] / plan['step_mm']))
    if profile is None:
        return planned
    covered = len(profile['residual_A']) - 1
    if covered not in (planned, planned-1):
        raise BaselineError('baseline must cover the full plan or end at most one measured step short')
    return covered


def load_profile(path, expected, allow_candidate=False):
    p = json.loads(Path(path).read_text())
    if p.get('kind') != 't7_empty_air_baseline' or p.get('version') != 1:
        raise BaselineError('unsupported baseline format')
    if p.get('binding') != expected:
        raise BaselineError('baseline plan/speed/setup binding does not match this run')
    core = {k: v for k, v in p.items() if k not in ('profile_sha256', 'qualification')}
    if p.get('profile_sha256') != digest(core):
        raise BaselineError('baseline contents changed since build')
    k = np.asarray(p.get('waypoint_index'), float)
    b = np.asarray(p.get('residual_A'), float)
    if k.ndim != 1 or len(k) < 2 or not np.array_equal(k, np.arange(len(k))):
        raise BaselineError('baseline must contain contiguous waypoints from zero')
    if b.shape != (len(k), 6) or not np.isfinite(b).all():
        raise BaselineError('invalid baseline currents')
    qualification = p.get('qualification', {})
    if not allow_candidate and (qualification.get('passed') is not True or
                                qualification.get('profile_sha256') != p['profile_sha256']):
        raise BaselineError('baseline is not independently qualified; use --air-validate with objects removed')
    return p


def expected_residual(profile, k):
    b = np.asarray(profile['residual_A'], float)
    if not isinstance(k, (int, np.integer)) or not 0 <= k < len(b):
        raise BaselineError(f'waypoint {k} outside calibrated range; extrapolation forbidden')
    return b[k]


def descent_reversal_audit(k, t, step_mm):
    """Bound cumulative feedback reversals, independent of how frames split them.

    The old per-frame one-waypoint test accepted arbitrarily long staircases
    backwards but rejected the same small excursion arriving in one frame.
    These are offline data-quality bounds, not a physical motion guarantee.
    """
    k, t = np.asarray(k), np.asarray(t)
    peak, started = k[0], None
    max_depth, max_duration, count = 0.0, 0.0, 0
    for i in range(len(k)):
        if k[i] >= peak:
            if started is not None:
                duration = float(t[i] - t[started])
                max_duration = max(max_duration, duration)
                if duration > .1 + 1e-9:
                    raise BaselineError('feedback reversal takes more than 0.1 s to recover')
                started = None
            peak = k[i]
        else:
            if started is None:
                started = i
                count += 1
            depth = float((peak - k[i]) * step_mm)
            max_depth = max(max_depth, depth)
            if depth > .5 + 1e-9:
                raise BaselineError('cumulative feedback reversal exceeds 0.5 mm')
            if t[i] - t[started] > .1 + 1e-9:
                raise BaselineError('feedback reversal takes more than 0.1 s to recover')
    if started is not None:
        raise BaselineError('feedback reversal has not recovered at descent end')
    return {'policy': 'cumulative_reversal_0p5mm_recovery_0p1s_v1',
            'max_cumulative_reversal_mm': max_depth, 'max_recovery_s': max_duration,
            'reversal_count': count}


def pre_descent_data(path, tr, plan, require_full_coverage=True, endpoint_tolerance_mm=None):
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    n = len(d['t'])
    if tr.get('n_rows') != n:
        raise BaselineError(f'{path}: summary/data row-count mismatch')
    for key in ('t', 'q', 'current', 'resid', 'k_now', 'state'):
        if not np.isfinite(d[key]).all():
            raise BaselineError(f'{path}: nonfinite {key}')
    if np.any(np.diff(d['t']) < 0) or np.max(np.diff(d['t'])) > .2:
        raise BaselineError(f'{path}: interrupted sampling')
    mask = d['state'] <= 2  # exclude the upward return, even when it revisits the same positions
    if np.any(np.diff(d['t'][mask]) <= 0):
        raise BaselineError(f'{path}: non-increasing descent timestamps')
    k = d['k_now'][mask].astype(int)
    tolerance_mm = (0 if require_full_coverage else
                    (plan['step_mm'] if endpoint_tolerance_mm is None else endpoint_tolerance_mm))
    if k[0] != 0 or (endpoint(plan)-k[-1])*plan['step_mm'] > tolerance_mm + 1e-9:
        raise BaselineError(f'{path}: incomplete descent')
    try:
        descent_reversal_audit(k, d['t'][mask], plan['step_mm'])
    except BaselineError as exc:
        raise BaselineError(f'{path}: {exc}') from exc
    if not require_full_coverage:
        if not baseline_endpoint_reached(endpoint(plan), k[-1], endpoint(plan),
                                         d['q'][mask][-1], plan['q'], plan['step_mm'], tolerance_mm):
            raise BaselineError(f'{path}: final joints or index exceed endpoint tolerance')
    if np.any(k < 0) or np.any(k > endpoint(plan) + 1):
        raise BaselineError(f'{path}: invalid waypoint range')
    free = (k * plan['step_mm'] >= plan['arm_after_mm'] - 10) & (k * plan['step_mm'] <= plan['arm_after_mm'])
    if free.sum() < 20:
        raise BaselineError(f'{path}: insufficient free-window samples')
    r = d['resid'][mask]
    centred = r - r[free].mean(axis=0)  # each trial's static/friction offset is a nuisance parameter
    uniq = np.unique(k)
    if np.max(np.diff(uniq)) * plan['step_mm'] > 1.0:
        raise BaselineError(f'{path}: gap larger than 1 mm in calibration data')
    values = np.array([np.median(centred[k == i], axis=0) for i in uniq])
    # Qualification may terminate within the versioned endpoint tolerance. Do not
    # extrapolate even this unused diagnostic curve. Training always covers all nodes.
    grid = np.arange((endpoint(plan) if require_full_coverage else min(endpoint(plan), k[-1])) + 1)
    if uniq[0] > grid[0] or uniq[-1] < grid[-1]:
        raise BaselineError('calibration does not cover requested path')
    curve = np.column_stack([np.interp(grid, uniq, values[:, j]) for j in range(6)])
    return curve, d


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def build(runs, out):
    curves, sources, hashes, records, ends = [], [], set(), [], []
    expected = plan = None
    for run in map(Path, runs):
        s = json.loads((run / 'summary.json').read_text())
        this_plan = json.loads((run / 'plan_snapshot.json').read_text())
        if s.get('air_mode') != 'calibration' or not s.get('empty_workspace_confirmed'):
            raise BaselineError(f'{run}: not an explicitly confirmed empty-air calibration')
        if s.get('air_clearance_mm', 0) < this_plan['max_depth_mm'] + 30:
            raise BaselineError(f'{run}: insufficient declared clearance')
        bind = binding(this_plan, s['speed_mm_s'], s['speed_pct'], s['baseline_context'])
        if expected is None:
            expected, plan = bind, this_plan
        if bind != expected:
            raise BaselineError('all calibrations must use the exact same plan, speed and setup')
        if (run / 'abort.json').exists():
            raise BaselineError(f'{run}: aborted calibration')
        for tr in s['trials']:
            if tr['status'] != 'air_complete' or tr.get('post_end') != 'still':
                raise BaselineError(f'{run}: calibration trial did not complete and retract')
            path = run / f"trial_{tr['trial']:02d}.npz"
            h = file_digest(path)
            if h in hashes:
                raise BaselineError('duplicate calibration trace')
            hashes.add(h)
            with np.load(path, allow_pickle=False) as d:
                measured = d['k_now'][d['state'] <= 2]
                if not len(measured) or not np.isfinite(measured).all():
                    raise BaselineError(f'{path}: missing descent positions')
                ends.append(int(measured[-1]))
            records.append((path, tr))
            sources.append({'run': str(run.resolve()), 'trial': tr['trial'], 'npz_sha256': h})
    if len(records) < 3:
        raise BaselineError('at least three complete, distinct empty-air trials are required')
    covered = min(endpoint(plan), min(ends))
    if covered < endpoint(plan)-1:
        raise BaselineError('calibration stopped more than one step short; incomplete coverage')
    # Legacy completion can occur at max_depth-step. Use only the common observed
    # prefix and bind later commands to that shorter endpoint; never pad/extrapolate.
    coverage_plan = dict(plan, max_depth_mm=covered*plan['step_mm'])
    for path, tr in records:
        curve, _ = pre_descent_data(path, tr, coverage_plan)
        curves.append(curve)
    for source, (path, _) in zip(sources, records):
        with np.load(path, allow_pickle=False) as d:
            mask = d['state'] <= 2
            source['reversal_audit'] = descent_reversal_audit(
                d['k_now'][mask], d['t'][mask], plan['step_mm'])
    b = np.median(curves, axis=0)
    # Fixed 1 mm spatial smoothing; never fitted on the qualification or contact trials.
    radius = max(1, round(.5 / plan['step_mm']))
    b = np.array([b[max(0, i-radius):min(len(b), i+radius+1)].mean(axis=0) for i in range(len(b))])
    p = {'kind': 't7_empty_air_baseline', 'version': 1, 'binding': expected,
         'waypoint_index': list(range(len(b))), 'residual_A': b.tolist(),
         'training_spread_A': np.std(curves, axis=0).tolist(), 'sources': sources,
         'planned_endpoint_index': endpoint(plan), 'covered_endpoint_index': covered,
         'covered_depth_mm': covered*plan['step_mm'],
         'correction': 'raw gravity residual minus spatial baseline; fixed per-run offset from the free window; no phi scaling',
         'validity': 'same powered session, unloaded closed gripper, plan, speed and setup; operator must reconfirm'}
    p['profile_sha256'] = digest(p)
    write_new(out, p)
    return p


def qualify(profile_path, run, out):
    run = Path(run)
    s = json.loads((run / 'summary.json').read_text())
    plan = json.loads((run / 'plan_snapshot.json').read_text())
    expected = binding(plan, s['speed_mm_s'], s['speed_pct'], s['baseline_context'])
    p = load_profile(profile_path, expected, allow_candidate=True)
    if (s.get('air_mode') != 'validation' or not s.get('empty_workspace_confirmed') or
            s.get('air_clearance_mm', 0) < plan['max_depth_mm'] + 30 or
            s.get('baseline_profile_sha256') != p['profile_sha256'] or (run/'abort.json').exists()):
        raise BaselineError('not an independent empty-air validation of this profile')
    if len(s['trials']) != 1:
        raise BaselineError('qualification requires one dedicated held-out trial')
    tr = s['trials'][0]
    if tr['status'] != 'air_validated' or tr.get('post_end') != 'still':
        raise BaselineError('validation stopped early or failed to retract; profile remains unqualified')
    path = run / f"trial_{tr['trial']:02d}.npz"
    h = file_digest(path)
    if h in {x['npz_sha256'] for x in p['sources']}:
        raise BaselineError('validation trace was used for calibration')
    effective_end = endpoint(plan, p)
    coverage_plan = dict(plan, max_depth_mm=effective_end*plan['step_mm'])
    policy = tr.get('endpoint_policy')
    if policy not in (None, ENDPOINT_POLICY):
        raise BaselineError('unknown endpoint policy')
    tolerance_mm = ENDPOINT_LAG_MM if policy == ENDPOINT_POLICY else plan['step_mm']
    _, d = pre_descent_data(path, tr, coverage_plan, require_full_coverage=False,
                           endpoint_tolerance_mm=tolerance_mm)
    if policy == ENDPOINT_POLICY:
        # Require logged command evidence, not merely an asserted success summary.
        row = np.flatnonzero(d['state'] <= 2)[-1]
        cmd = int(d['cmd_idx'][row])
        if (d['state'][row] != 2 or not 0 <= cmd < len(d['cmd_val'])
                or d['cmd_kind'][cmd] != 1 or d['cmd_t'][cmd] > d['t_fb'][row]
                or not np.allclose(d['cmd_val'][cmd], plan['q'][effective_end], rtol=0, atol=1e-10)
                or not baseline_endpoint_reached(d['k_cmd'][row], d['k_now'][row], effective_end,
                                                  d['q'][row], plan['q'], plan['step_mm'])):
            raise BaselineError('validation lacks final command / endpoint evidence')
    mask = (d['state'] >= 1) & (d['state'] <= 2) & (d['fz_depth_mm'] >= plan['arm_after_mm'] + 3)
    f = d['tip_force_mean_N'][mask]
    if len(f) < 20 or not np.isfinite(f).all():
        raise BaselineError('validation force record is incomplete')
    # Two-sided margin: a negative bias can conceal real contact just as a positive bias causes false alarms.
    max_abs = float(np.max(np.abs(f)))
    if max_abs > 1.5:
        raise BaselineError(f'empty-air mean force reaches {max_abs:.3f} N; needs |F| <= 1.5 N for margin to the 2 N floor')
    p['qualification'] = {'passed': True, 'profile_sha256': p['profile_sha256'],
                          'run': str(run.resolve()), 'npz_sha256': h,
                          'max_abs_mean_force_N': max_abs, 'limit_N': 1.5,
                          'validated_end_depth_mm': float(d['fz_depth_mm'][d['state'] <= 2][-1]),
                          'command_limit_mm': effective_end*plan['step_mm'],
                          'endpoint_tolerance_mm': tolerance_mm,
                          'endpoint_joint_tolerance_deg': ENDPOINT_JOINT_DEG,
                          'endpoint_policy': policy or 'legacy_one_waypoint'}
    write_new(out, p)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--runs', nargs='+', required=True)
    b.add_argument('--out', required=True)
    q = sub.add_parser('qualify')
    q.add_argument('--profile', required=True)
    q.add_argument('--run', required=True)
    q.add_argument('--out', required=True)
    args = ap.parse_args()
    try:
        p = build(args.runs, args.out) if args.command == 'build' else qualify(args.profile, args.run, args.out)
    except (BaselineError, KeyError, ValueError) as exc:
        raise SystemExit(str(exc))
    print(f"saved {args.out}; qualified={p.get('qualification', {}).get('passed', False)}; profile={p['profile_sha256']}")


if __name__ == '__main__':
    main()

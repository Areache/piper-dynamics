"""Offline E analysis of measured tip points in one robot-base frame, in mm.

Input CSV: phase,point,trial,x_mm,y_mm,z_mm (flat/tilt, four point IDs, trials 0..2).
Coordinates must be extracted from actual joint feedback by the matching FK model.
No robot connection; no force recomputation; detection and onset proxies stay separate.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def fit_plane(points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError('need at least three finite XYZ points in mm')
    center = p.mean(axis=0)
    xy = p[:, :2] - center[:2]
    sv = np.linalg.svd(xy, compute_uv=False)
    if sv[0] <= 0 or sv[1] <= sv[0]*1e-6:
        raise ValueError('point XY coordinates are collinear or degenerate')
    a, b = np.linalg.lstsq(xy, p[:, 2]-center[2], rcond=None)[0]
    c = center[2] - a*center[0] - b*center[1]
    coef = np.array([a, b, c])
    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    residual = p[:, 2] - np.c_[p[:, :2], np.ones(len(p))] @ coef
    return {'z_equals_ax_plus_by_plus_c': coef.tolist(), 'up_normal': normal.tolist(),
            'inclination_deg': float(np.degrees(np.arctan(np.hypot(a, b)))),
            'xy_condition_number': float(sv[0]/sv[1]),
            'vertical_residual_mm': residual.tolist(),
            'vertical_rms_mm': float(np.sqrt(np.mean(residual**2)))}


def normal_angle_deg(n0, n1):
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(n0, n1)), np.dot(n0, n1))))


def phase_report(groups):
    ids = sorted(groups)
    means = np.array([np.mean(groups[i], axis=0) for i in ids])
    full = fit_plane(means)
    holdouts = []
    for i, name in enumerate(ids):
        fit = fit_plane(np.delete(means, i, axis=0))
        coef = fit['z_equals_ax_plus_by_plus_c']
        e = float(means[i, 2] - np.dot([*means[i, :2], 1.0], coef))
        holdouts.append({'fitted_points': [x for x in ids if x != name], 'held_out_point': name,
                         'vertical_error_mm': e,
                         'perpendicular_error_mm': e*fit['up_normal'][2], 'plane': fit})
    pair_distances = [float(np.linalg.norm(means[i, :2]-means[j, :2]))
                      for i in range(4) for j in range(i+1, 4)]
    return {'points': {name: {'mean_xyz_mm': means[i].tolist(),
                              'sample_sd_xyz_mm': np.std(groups[name], axis=0, ddof=1).tolist(),
                              'trials': 3} for i, name in enumerate(ids)},
            'min_xy_spacing_mm': min(pair_distances),
            'readme_100mm_spacing_met': min(pair_distances) >= 100,
            'four_point_plane': full, 'three_point_holdouts': holdouts,
            'max_abs_held_out_vertical_error_mm': max(abs(h['vertical_error_mm']) for h in holdouts)}


def analyze(rows, coordinate_kind, rise_mm=None, span_horizontal_mm=None):
    if coordinate_kind not in ('detection', 'force_onset_proxy'):
        raise ValueError('coordinate kind must distinguish detection from an onset proxy')
    groups = {'flat': {}, 'tilt': {}}
    seen = set()
    for row in rows:
        phase, point = row['phase'], str(row['point']).strip()
        trial = str(row['trial'])
        if phase not in groups or not point or trial not in ('0', '1', '2'):
            raise ValueError('expected flat/tilt, nonempty point ID, trial 0, 1 or 2')
        key = (phase, point, trial)
        if key in seen:
            raise ValueError(f'duplicate point/trial: {key}')
        seen.add(key)
        xyz = np.array([float(row[k]) for k in ('x_mm', 'y_mm', 'z_mm')])
        if not np.isfinite(xyz).all():
            raise ValueError('nonfinite point coordinates')
        groups[phase].setdefault(point, []).append(xyz)
    if (len(groups['flat']) != 4 or set(groups['flat']) != set(groups['tilt']) or
            any(len(v) != 3 for phase in groups.values() for v in phase.values())):
        raise ValueError('need the same four point IDs and three distinct trials per point in each phase')
    phases = {name: phase_report(g) for name, g in groups.items()}
    flat = phases['flat']['four_point_plane']; tilt = phases['tilt']['four_point_plane']
    rotation = normal_angle_deg(flat['up_normal'], tilt['up_normal'])
    report = {'coordinate_frame': 'robot base, XYZ in mm', 'coordinate_kind': coordinate_kind,
              'phases': phases, 'normal_rotation_deg': rotation,
              'inclination_difference_deg': tilt['inclination_deg']-flat['inclination_deg'],
              'xy_shift_between_phase_means_mm': {
                  p: float(np.linalg.norm(np.array(phases['tilt']['points'][p]['mean_xyz_mm'])[:2] -
                                          np.array(phases['flat']['points'][p]['mean_xyz_mm'])[:2]))
                  for p in groups['flat']},
              'acceptance': 'measurement report only; no blanket pass without declared tolerances and operator observations',
              'reference': None}
    if (rise_mm is None) != (span_horizontal_mm is None):
        raise ValueError('rise and horizontal support spacing must be supplied together')
    if rise_mm is not None:
        if not np.isfinite([rise_mm, span_horizontal_mm]).all() or rise_mm < 0 or span_horizontal_mm <= 0:
            raise ValueError('reference rise must be finite >=0 and horizontal spacing finite >0')
        angle = float(np.degrees(np.arctan2(rise_mm, span_horizontal_mm)))
        report['reference'] = {'rise_mm': rise_mm, 'horizontal_support_spacing_mm': span_horizontal_mm,
                               'expected_angle_deg': angle, 'normal_rotation_error_deg': rotation-angle,
                               'assumptions': 'initial physical plate level; rise is actual vertical height difference; '
                                              'L is horizontal separation perpendicular to support lines, not slanted board length'}
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--points', type=Path, required=True)
    ap.add_argument('--coordinate-kind', choices=['detection', 'force_onset_proxy'], required=True)
    ap.add_argument('--rise-mm', type=float)
    ap.add_argument('--span-horizontal-mm', type=float)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    with a.points.open(newline='') as f:
        result = analyze(list(csv.DictReader(f)), a.coordinate_kind, a.rise_mm, a.span_horizontal_mm)
    with a.out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')
    print(f'saved {a.out}; normal rotation {result["normal_rotation_deg"]:.4f} deg')


if __name__ == '__main__':
    main()

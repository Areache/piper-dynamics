"""Synthetic geometric ground truth; no robot or SDK imports."""
import unittest

import numpy as np

from e_plane_analysis import analyze, fit_plane, normal_angle_deg


def samples(flat_slope=0., tilt_slope=.1):
    rows = []
    for phase, slope in [('flat', flat_slope), ('tilt', tilt_slope)]:
        for point, (x, y) in zip('ABCD', [(230, -60), (350, -60), (230, 90), (350, 90)]):
            for trial, noise in enumerate([-.1, 0., .1]):
                rows.append(dict(phase=phase, point=point, trial=trial,
                                 x_mm=x, y_mm=y, z_mm=slope*(x-230)+20+noise))
    return rows


class PlaneTests(unittest.TestCase):
    def test_known_plane_and_independent_fourth_point(self):
        r = analyze(samples(), 'detection', 30, 300)
        self.assertAlmostEqual(r['normal_rotation_deg'], np.degrees(np.arctan(.1)))
        self.assertAlmostEqual(r['reference']['normal_rotation_error_deg'], 0.)
        self.assertAlmostEqual(r['phases']['tilt']['max_abs_held_out_vertical_error_mm'], 0.)
        self.assertTrue(r['phases']['flat']['readme_100mm_spacing_met'])
        self.assertAlmostEqual(r['phases']['flat']['points']['A']['sample_sd_xyz_mm'][2], .1)

    def test_fourth_point_is_not_used_to_fit_its_plane(self):
        rows = samples()
        for row in rows:
            if row['phase'] == 'tilt' and row['point'] == 'D':
                row['z_mm'] += 3
        r = analyze(rows, 'detection')
        held = next(x for x in r['phases']['tilt']['three_point_holdouts'] if x['held_out_point'] == 'D')
        self.assertAlmostEqual(held['vertical_error_mm'], 3.)
        self.assertAlmostEqual(r['phases']['tilt']['four_point_plane']['vertical_rms_mm'], .75)

    def test_rotation_is_not_difference_of_inclinations(self):
        slope = np.tan(np.radians(10))
        r = analyze(samples(-slope, slope), 'force_onset_proxy')
        self.assertAlmostEqual(r['inclination_difference_deg'], 0.)
        self.assertAlmostEqual(r['normal_rotation_deg'], 20.)

    def test_geometry_and_missing_duplicate_nonfinite_data_rejected(self):
        with self.assertRaises(ValueError):
            fit_plane([[0, 0, 1], [1, 1, 2], [2, 2, 3]])
        for rows in [samples()[:-1], samples()+[samples()[0]],
                     [dict(r, z_mm=float('nan')) if i == 0 else r for i, r in enumerate(samples())]]:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                analyze(rows, 'detection')
        with self.assertRaises(ValueError):
            analyze(samples(), 'detection', 30, None)
        with self.assertRaises(ValueError):
            analyze(samples(), 'detection', 30, 0)


if __name__ == '__main__':
    unittest.main()

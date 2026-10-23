"""Read-only recorder rejects corrupt evidence; no hardware connection."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import t7_record_start as R


class RecorderTests(unittest.TestCase):
    def test_zero_joints_are_valid_and_nonfinite_data_is_not(self):
        now = [0.]
        def clock(): return now[0]
        def sleep(dt): now[0] += dt
        robot = SimpleNamespace()
        with patch.object(R, 'wait_for_feedback'), patch.object(R, 'read_state', return_value=(np.zeros(6),)*4):
            q, sd = R.sample_pose(robot, clock, sleep)
            self.assertTrue(np.isfinite(q).all())
            self.assertEqual(float(sd.max()), 0.)
        now[0] = 0.
        with patch.object(R, 'wait_for_feedback'), patch.object(R, 'read_state', return_value=(np.full(6, np.nan),)*4):
            with self.assertRaises(R.OffPlanFeedback):
                R.sample_pose(robot, clock, sleep)

    def test_moving_samples_cannot_be_kept_as_a_start(self):
        now = [0.]
        def sleep(dt): now[0] += dt
        def state(*args):
            return (np.full(6, now[0]*.1), np.zeros(6), np.zeros(6), np.zeros(6))
        with patch.object(R, 'wait_for_feedback'), patch.object(R, 'read_state', side_effect=state):
            with self.assertRaisesRegex(ValueError, 'moved'):
                R.sample_pose(SimpleNamespace(), lambda: now[0], sleep)

    def test_gap_rejects_nan_infinity_and_nonpositive_values(self):
        self.assertEqual(R.positive_gap('80.5'), 80.5)
        for value in ['nan', 'inf', '-inf', '0', '-10', 'bad']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                R.positive_gap(value)


if __name__ == '__main__':
    unittest.main()

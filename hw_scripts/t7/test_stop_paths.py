"""Offline stop-path regression tests. No SDK import or hardware connections."""
import contextlib
import io
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import guarded_move as G

HERE = Path(__file__).resolve().parent
PLAN_FILE = HERE / 'plans/right_p0.json'
if not PLAN_FILE.exists():   # plans/ is gitignored: these tests need locally generated plan JSONs
    raise unittest.SkipTest(f'needs {PLAN_FILE.name} (run t7_plan.py to create it)')
PLAN = json.loads((HERE / 'plans/right_p0.json').read_text())


class FakeArm:
    def __init__(self):
        self.q = np.asarray(PLAN['q'][10]) + 0.0001
        self.commands = []
        self.t = 0.0
        self.interrupt_at = None
        self.estops = 0

    def get_joint_angles(self):
        return SimpleNamespace(msg=self.q.copy())

    def get_motor_states(self, i):
        return SimpleNamespace(msg=SimpleNamespace(position=self.q[i - 1], velocity=0.0, current=0.0))

    def get_flange_pose(self):
        return SimpleNamespace(msg=[0.3, 0.0, 0.2, 0.0, 0.0, 0.0])

    def move_j(self, q):
        self.commands.append(('j', list(q)))

    def move_l(self, pose):
        self.commands.append(('l', list(pose)))

    def electronic_emergency_stop(self):
        self.estops += 1

    def set_speed_percent(self, pct):
        pass

    def clock(self):
        return self.t

    def sleep(self, dt):
        self.t += dt
        if self.interrupt_at is not None and self.t >= self.interrupt_at:
            self.interrupt_at = None
            raise KeyboardInterrupt


def trial_result(status, overshoot=0.0):
    data = {k: np.zeros(2) for k in ('t', 'fz_phase', 'fz_depth_mm', 'fz_tip_z_mm',
            'fz_cmd_depth_mm', 'fz_F_N', 'fz_F_mean_N', 'fz_phi')}
    return data, {'status': status, 'k_det': None, 'overshoot_mm': overshoot}


class StopPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.robot = FakeArm()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(patch.stopall)
        patch.object(G, 'connect', side_effect=AssertionError('hardware forbidden')).start()
        patch.object(G, 'record_static', return_value={'current': np.zeros((3, 6))}).start()

    def run_main(self, name='run', trials=2):
        out = self.root / name
        with contextlib.redirect_stdout(io.StringIO()):
            G.main(['--plan', str(HERE / 'plans/right_p0.json'), '--trials', str(trials),
                    '--operator-confirmed', '--no-confirm-each', '--out', str(out)],
                   robot=self.robot, clock=self.robot.clock, sleep=self.robot.sleep,
                   ask=lambda _: 'YES')
        return out

    def test_protection_stops_never_return_or_repeat(self):
        for status in ('interrupted', 'timeout', 'stall', 'contact_late', 'too_fast', 'max_depth', 'no_contact'):
            with self.subTest(status=status), patch.object(G, 'settle') as settle, \
                    patch.object(G, 'descend', side_effect=lambda *a, **k: trial_result(status)) as descend:
                out = self.run_main(status)
                self.assertEqual(settle.call_count, 1, 'only initial approach is permitted')
                self.assertEqual(descend.call_count, 1)
                self.assertEqual(len(json.loads((out / 'summary.json').read_text())['trials']), 1)
                self.assertTrue((out / 'trial_00.json').exists())
                self.assertTrue((out / 'fz_00.csv').exists())

    def test_successful_contact_and_test_stop_still_return(self):
        for status in ('contact', 'test_stop'):
            with self.subTest(status=status), patch.object(G, 'settle') as settle, \
                    patch.object(G, 'descend', side_effect=lambda *a, **k: trial_result(status)) as descend:
                self.run_main(status)
                self.assertEqual(settle.call_count, 3)  # approach and two normal returns
                self.assertEqual(descend.call_count, 2)

    def test_excess_overshoot_stops_before_return(self):
        for status in ('contact', 'test_stop'):
            with self.subTest(status=status), patch.object(G, 'settle') as settle, \
                    patch.object(G, 'descend', side_effect=lambda *a, **k: trial_result(status, 3.25)) as descend:
                self.run_main(status)
                self.assertEqual(settle.call_count, 1)
                self.assertEqual(descend.call_count, 1)

    def test_outer_interrupt_and_timeout_hold_and_preserve_abort(self):
        for label, exc in [('interrupt', KeyboardInterrupt()), ('timeout', TimeoutError('settle failed'))]:
            for stage in ('approach', 'return', 'descent_or_post'):
                with self.subTest(label=label, stage=stage):
                    self.robot.commands.clear()
                    settle_effect = [exc] if stage == 'approach' else ([None, exc] if stage == 'return' else None)
                    with patch.object(G, 'settle', side_effect=settle_effect), \
                            patch.object(G, 'descend', side_effect=exc if stage == 'descent_or_post'
                                         else lambda *a, **k: trial_result('contact')):
                        name = label + stage
                        with self.assertRaises(type(exc)):
                            self.run_main(name)
                        abort = json.loads((self.root / name / 'abort.json').read_text())
                        self.assertEqual(abort['status'], 'interrupted' if label == 'interrupt' else 'timeout')
                        np.testing.assert_array_equal(self.robot.commands[-1][1], self.robot.q)
                        np.testing.assert_array_equal(abort['hold_q_rad'], self.robot.q)

    def test_invalid_feedback_uses_damped_estop_without_joint_command(self):
        for q in (np.full(6, np.nan), np.zeros(5), np.full(6, np.inf)):
            with self.subTest(q=q):
                self.robot.q = q
                with self.assertRaises(RuntimeError):
                    G.hold_measured_position(self.robot)
                self.assertFalse(self.robot.commands)
        self.assertEqual(self.robot.estops, 3)

    def test_feedback_exception_uses_damped_estop(self):
        with patch.object(self.robot, 'get_joint_angles', side_effect=OSError('read failure')):
            with self.assertRaises(OSError):
                G.hold_measured_position(self.robot)
        self.assertEqual(self.robot.estops, 1)
        self.assertFalse(self.robot.commands)

    def test_real_descent_timeout_holds_exact_pose_in_both_modes(self):
        for mode in ('stream', 'line'):
            with self.subTest(mode=mode):
                self.robot = FakeArm()  # intentionally does not follow commands
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)  # no force before arming
                    data, summary = G.descend(self.robot, PLAN, 10.0,
                        self.robot.clock, self.robot.sleep, motion=mode, log=lambda _: None)
                self.assertEqual(summary['status'], 'timeout')
                holds = np.flatnonzero(data['cmd_kind'] == 5)
                self.assertEqual(len(holds), 1)
                for q in data['cmd_val'][holds[0]:]:
                    np.testing.assert_array_equal(q, self.robot.q)

    def test_real_descent_interrupt_holds_exact_pose(self):
        self.robot.interrupt_at = 0.15
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            data, summary = G.descend(self.robot, PLAN, 10.0, self.robot.clock,
                                     self.robot.sleep, log=lambda _: None)
        self.assertEqual(summary['status'], 'interrupted')
        holds = np.flatnonzero(data['cmd_kind'] == 5)
        self.assertEqual(len(holds), 1)
        for q in data['cmd_val'][holds[0]:]:
            np.testing.assert_array_equal(q, self.robot.q)


    def test_startup_rejects_missing_joint_pair_without_commands(self):
        self.robot._parser = SimpleNamespace(joint_12=SimpleNamespace(timestamp=1000),
                                            joint_34=None, joint_56=SimpleNamespace(timestamp=1000))
        with patch.object(G.time, 'time', return_value=1000), patch.object(G, 'settle') as settle:
            with self.assertRaises(G.FeedbackNotReady):
                self.run_main('missing_pair')
        settle.assert_not_called()
        self.assertFalse(self.robot.commands)
        self.assertEqual(self.robot.estops, 0)

    def test_startup_rejects_stale_pairs(self):
        self.robot._parser = SimpleNamespace(**{k: SimpleNamespace(timestamp=999)
            for k in ('joint_12', 'joint_34', 'joint_56')})
        with patch.object(G.time, 'time', return_value=1000):
            with self.assertRaises(G.FeedbackNotReady):
                G.wait_for_feedback(self.robot, self.robot.clock, self.robot.sleep)
        self.assertFalse(self.robot.commands)

    def test_complete_zero_pose_is_valid(self):
        self.robot.q = np.zeros(6)
        self.robot._parser = SimpleNamespace(**{k: SimpleNamespace(timestamp=1000)
            for k in ('joint_12', 'joint_34', 'joint_56')})
        with patch.object(G.time, 'time', return_value=1000):
            state = G.wait_for_feedback(self.robot, self.robot.clock, self.robot.sleep)
        np.testing.assert_array_equal(state[0], np.zeros(6))
        self.assertGreaterEqual(self.robot.t, 0.25)
        self.assertFalse(self.robot.commands)

    def test_inconsistent_sample_does_not_become_endpoint_depth(self):
        Q = np.asarray(PLAN['q'])
        q = Q[0].copy()
        q[2:4] = [0.03858923, 0.57218874]  # actual anomalous sample from failed observer
        with self.assertRaises(G.OffPlanFeedback):
            G.locate_on_plan(Q, q, 0, len(Q))

    def test_off_plan_feedback_estops_without_return(self):
        with patch.object(G, 'settle') as settle, patch.object(G, 'descend',
                side_effect=G.OffPlanFeedback('bad sample')):
            with self.assertRaises(G.OffPlanFeedback):
                self.run_main('off_plan')
            self.assertEqual(settle.call_count, 1)
        self.assertEqual(self.robot.estops, 1)
        abort = json.loads((self.root / 'off_plan/abort.json').read_text())
        self.assertEqual(abort['status'], 'invalid_feedback')


if __name__ == '__main__':
    unittest.main()

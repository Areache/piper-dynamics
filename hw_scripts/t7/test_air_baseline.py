"""Offline end-to-end tests. Fake robot only; no CAN/SDK imports."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import air_baseline as B
import guarded_move as G

HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE / 'plans/white_table_plastic132p4_lift45.json'
PLAN_FILE = PLAN_PATH
if not PLAN_FILE.exists():   # plans/ is gitignored: these tests need locally generated plan JSONs
    raise unittest.SkipTest(f'needs {PLAN_FILE.name} (run t7_plan.py to create it)')
PLAN = json.loads(PLAN_PATH.read_text())
Q = np.array(PLAN['q'])
IE = np.array(PLAN['i_expected_A'])
A = np.array(PLAN['di_per_N_up'])


class SimArm:
    """Perfect position tracking; repeatable J2 reversal-current drift plus deterministic noise."""
    def __init__(self, seed=0, external_force=0., force_depth=40., stall_depth=None, interrupt_depth=None):
        self.q = Q[0].copy()
        self.k = 0
        self.t = 0.
        self.seed = seed
        self.external_force = external_force
        self.force_depth = force_depth
        self.stall_depth = stall_depth
        self.interrupt_depth = interrupt_depth
        self.commands = []
        self.estops = 0

    def clock(self):
        return self.t

    def sleep(self, dt):
        self.t += dt
        if self.interrupt_depth is not None and self.k*PLAN['step_mm'] >= self.interrupt_depth:
            self.interrupt_depth = None
            raise KeyboardInterrupt

    def move_j(self, q):
        self.commands.append(np.array(q))
        k = int(np.argmin(np.max(np.abs(Q-np.array(q)), axis=1)))
        if self.stall_depth is None or k*PLAN['step_mm'] <= self.stall_depth:
            self.k = k
            self.q = np.array(q)

    def get_joint_angles(self):
        return SimpleNamespace(msg=self.q.copy())

    def get_motor_states(self, i):
        j = i-1
        depth = self.k*PLAN['step_mm']
        current = IE[self.k, j] + .01*self.seed + .007*np.sin(37*self.t + i + self.seed)
        if self.k > 0:
            current += [.0, .10, .03, 0, 0, 0][j]
            if j == 1:
                current += max(0., depth-25.)*.025
        if depth >= self.force_depth:
            current += self.external_force*A[self.k, j]
        return SimpleNamespace(msg=SimpleNamespace(position=self.q[j], velocity=0., current=current))

    def set_speed_percent(self, pct):
        pass

    def electronic_emergency_stop(self):
        self.estops += 1


class AirBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.patch = patch.object(G, 'connect', side_effect=AssertionError('hardware forbidden'))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def run_sim(self, name, flags, arm=None, trials=1, plan_path=PLAN_PATH):
        arm = arm or SimArm()
        out = self.root/name
        argv = ['--plan', str(plan_path), '--operator-confirmed', '--out', str(out),
                '--baseline-context', 'offline_test', '--trials', str(trials), *flags]
        with contextlib.redirect_stdout(io.StringIO()):
            G.main(argv, robot=arm, clock=arm.clock, sleep=arm.sleep,
                   ask=lambda prompt: 'AIR CLEAR' if 'AIR CLEAR' in prompt else ('YES' if 'YES' in prompt else 'y'))
        return out, arm

    def make_candidate(self):
        cal, _ = self.run_sim('cal', ['--air-calibrate', '--air-clearance-mm', '100'], trials=3)
        candidate = self.root/'candidate.json'
        p = B.build([cal], candidate)
        return candidate, p

    def make_qualified(self):
        candidate, _ = self.make_candidate()
        val, _ = self.run_sim('validation', ['--air-validate', '--air-clearance-mm', '100', '--air-baseline', str(candidate)], SimArm(seed=7))
        qualified = self.root/'qualified.json'
        p = B.qualify(candidate, val, qualified)
        return qualified, p

    def test_end_to_end_calibration_independent_validation_and_real_contact(self):
        qualified, p = self.make_qualified()
        self.assertTrue(p['qualification']['passed'])
        run, _ = self.run_sim('contact', ['--air-baseline', str(qualified)], SimArm(seed=8, external_force=3.5))
        s = json.loads((run/'summary.json').read_text())['trials'][0]
        self.assertEqual(s['status'], 'contact')
        self.assertGreaterEqual(s['det_depth_mm'], 40.)
        self.assertLess(s['det_depth_mm'], 43.)
        d = np.load(run/'trial_00.npz')
        self.assertTrue(np.isnan(d['fz_F_N'][d['state'] >= 3]).all(), 'no downward compensation during retract')
        mask = (d['state'] >= 1) & (d['state'] <= 2) & (d['fz_depth_mm'] >= 23)
        np.testing.assert_allclose(d['fz_F_mean_N'][mask], d['tip_force_mean_N'][mask], atol=.1)

    def test_legacy_drift_reproduces_false_positive(self):
        arm = SimArm()
        data, s = G.descend(arm, PLAN, 10., arm.clock, arm.sleep, log=lambda _: None)
        self.assertEqual(s['status'], 'contact')
        self.assertLess(s['det_depth_mm'], 50.)

    def test_runtime_cap_stops_commands_without_changing_plan_or_profile(self):
        qualified, p = self.make_qualified()
        run, arm = self.run_sim('capped_air', ['--air-baseline', str(qualified), '--max-depth-mm', '45.13'])
        s = json.loads((run/'summary.json').read_text())
        self.assertEqual(s['trials'][0]['status'], 'max_depth')
        self.assertEqual(s['requested_max_depth_mm'], 45.13)
        self.assertEqual(s['trials'][0]['commanded_endpoint_mm'], 45.)
        self.assertEqual(s['baseline_profile_sha256'], p['profile_sha256'])
        self.assertEqual(json.loads((run/'plan_snapshot.json').read_text()), PLAN)
        command_depths = [np.argmin(np.max(np.abs(Q-q), axis=1))*PLAN['step_mm'] for q in arm.commands]
        self.assertEqual(max(command_depths), 45.)
        self.assertEqual(command_depths[-1], 45., 'no return to start after no-contact limit')
        d = np.load(run/'trial_00.npz')
        self.assertLess(d['t'][-1], 12., 'no default 15-second bottom dwell')

    def test_runtime_cap_keeps_contact_detection_and_backoff(self):
        qualified, _ = self.make_qualified()
        run, arm = self.run_sim('capped_contact', ['--air-baseline', str(qualified), '--max-depth-mm', '45'],
                                SimArm(seed=8, external_force=3.5))
        s = json.loads((run/'summary.json').read_text())['trials'][0]
        self.assertEqual(s['status'], 'contact')
        self.assertLess(s['det_depth_mm'], 43.)
        self.assertEqual(s['post_end'], 'still')
        np.testing.assert_allclose(arm.q, Q[0])

    def test_runtime_cap_invalid_inputs_rejected_before_motion(self):
        qualified, _ = self.make_qualified()
        for value in ['nan', 'inf', '-1', '0', '23', '23.01', '60']:
            arm = SimArm()
            with self.subTest(value=value), self.assertRaises(SystemExit):
                self.run_sim('bad_cap', ['--air-baseline', str(qualified), '--max-depth-mm', value], arm)
            self.assertFalse(arm.commands)
        for flags in [[], ['--air-calibrate', '--air-clearance-mm', '100'],
                      ['--air-baseline', str(qualified), '--air-validate', '--air-clearance-mm', '100']]:
            arm = SimArm()
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.run_sim('bad_cap_mode', [*flags, '--max-depth-mm', '45'], arm)
            self.assertFalse(arm.commands)

    def test_candidate_rejected_before_motion_with_objects(self):
        candidate, _ = self.make_candidate()
        arm = SimArm()
        with self.assertRaises(SystemExit):
            self.run_sim('forbidden', ['--air-baseline', str(candidate)], arm)
        self.assertEqual(len(arm.commands), 0)

    def test_negative_bias_cannot_qualify(self):
        candidate, _ = self.make_candidate()
        val, _ = self.run_sim('negative', ['--air-validate', '--air-clearance-mm', '100', '--air-baseline', str(candidate)],
                              SimArm(seed=3, external_force=-3., force_depth=30.))
        self.assertEqual(json.loads((val/'summary.json').read_text())['trials'][0]['status'], 'air_validated')
        with self.assertRaisesRegex(B.BaselineError, 'needs'):
            B.qualify(candidate, val, self.root/'bad.json')

    def test_positive_bias_stops_validation_and_cannot_qualify(self):
        candidate, _ = self.make_candidate()
        val, _ = self.run_sim('positive', ['--air-validate', '--air-clearance-mm', '100', '--air-baseline', str(candidate)],
                              SimArm(seed=3, external_force=3., force_depth=30.))
        self.assertEqual(json.loads((val/'summary.json').read_text())['trials'][0]['status'], 'contact')
        with self.assertRaises(B.BaselineError):
            B.qualify(candidate, val, self.root/'bad.json')

    def test_context_speed_and_tampering_rejected(self):
        candidate, p = self.make_candidate()
        for key, value in [('context', 'new_power_session'), ('speed_mm_s', 9.), ('speed_pct', 15)]:
            expected = dict(p['binding']); expected[key] = value
            with self.subTest(key=key), self.assertRaises(B.BaselineError):
                B.load_profile(candidate, expected, allow_candidate=True)
        p['residual_A'][3][1] += .1
        candidate.write_text(json.dumps(p))
        with self.assertRaisesRegex(B.BaselineError, 'changed'):
            B.load_profile(candidate, p['binding'], allow_candidate=True)

    def test_no_extrapolation_and_no_partial_calibrations(self):
        candidate, p = self.make_candidate()
        for k in (-1, len(p['residual_A'])):
            with self.subTest(k=k), self.assertRaises(B.BaselineError):
                B.expected_residual(p, k)
        s_path = self.root/'cal'/'summary.json'
        s = json.loads(s_path.read_text());s['trials'][0]['status'] = 'contact'
        s_path.write_text(json.dumps(s))
        with self.assertRaises(B.BaselineError):
            B.build([self.root/'cal'], self.root/'partial.json')

    def test_duplicate_training_trace_rejected(self):
        candidate, _ = self.make_candidate()
        with self.assertRaisesRegex(B.BaselineError, 'duplicate'):
            B.build([self.root/'cal', self.root/'cal'], self.root/'duplicates.json')

    def test_reversal_audit_is_consistent_across_frame_splitting(self):
        # Both traces describe the same 0.5 mm excursion and prompt recovery.
        for k in ([100, 98, 100, 101], [100, 99, 98, 99, 100, 101]):
            with self.subTest(k=k):
                audit = B.descent_reversal_audit(k, np.arange(len(k))*.01, .25)
                self.assertEqual(audit['max_cumulative_reversal_mm'], .5)
                self.assertEqual(audit['reversal_count'], 1)

    def test_reversal_audit_rejects_staircase_and_sustained_or_unrecovered_retreat(self):
        cases = [([100, 99, 98, 97, 100], [0, .01, .02, .03, .04]),
                 ([100, 98, 98, 100], [0, .01, .08, .12]),
                 ([100, 99, 99, 99, 100], [0, .01, .12, .13, .14]),
                 ([100, 99], [0, .01])]
        for k, t in cases:
            with self.subTest(k=k, t=t), self.assertRaises(B.BaselineError):
                B.descent_reversal_audit(k, t, .25)

    def test_build_records_new_reversal_policy_and_still_rejects_bad_trace(self):
        cal, _ = self.run_sim('audit_cal', ['--air-calibrate', '--air-clearance-mm', '100'], trials=3)
        trace = cal/'trial_00.npz'
        with np.load(trace) as f:
            d = {k: f[k].copy() for k in f.files}
        i = np.flatnonzero((d['k_now'] > 100) & (d['state'] == 1))[0]
        d['k_now'][i] = d['k_now'][i-1] - 2
        d['q'][i] = Q[int(d['k_now'][i])]
        np.savez_compressed(trace, **d)
        p = B.build([cal], self.root/'audit_candidate.json')
        self.assertEqual(p['sources'][0]['reversal_audit']['max_cumulative_reversal_mm'], .5)
        # Three one-step retreats were accepted by the old per-frame guard.
        d['k_now'][i:i+3] = d['k_now'][i-1] - np.arange(1, 4)
        d['q'][i:i+3] = Q[d['k_now'][i:i+3].astype(int)]
        np.savez_compressed(trace, **d)
        with self.assertRaisesRegex(B.BaselineError, 'cumulative'):
            B.build([cal], self.root/'bad_staircase_candidate.json')

    def test_calibration_stall_and_interrupt_remain_protective(self):
        for name, kwargs, expected in [('stall', {'stall_depth': 30.}, 'stall'),
                                       ('interrupt', {'interrupt_depth': 30.}, 'interrupted')]:
            run, arm = self.run_sim(name, ['--air-calibrate', '--air-clearance-mm', '100'], SimArm(**kwargs), trials=3)
            trials = json.loads((run/'summary.json').read_text())['trials']
            self.assertEqual(len(trials), 1)
            self.assertEqual(trials[0]['status'], expected)
            self.assertGreater(np.max(np.abs(arm.q-Q[0])), .001, 'no automatic return to q0 after protection')

    def test_unsafe_cli_rejected_before_motion(self):
        for flags in [ ['--air-calibrate'], ['--air-calibrate', '--air-clearance-mm', '60'],
                       ['--air-calibrate', '--air-clearance-mm', '100', '--test-stop-mm', '50'],
                       ['--air-calibrate', '--air-clearance-mm', '100', '--motion', 'line'],
                       ['--air-calibrate', '--air-clearance-mm', '100', '--floor-n', '4'] ]:
            arm = SimArm()
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.run_sim('bad', flags, arm)
            self.assertFalse(arm.commands)

    def test_stale_motor_feedback_rejected(self):
        parser = SimpleNamespace(**{k: SimpleNamespace(timestamp=100.) for k in ('joint_12', 'joint_34', 'joint_56')})
        robot = SimpleNamespace(_parser=parser)
        with patch.object(G.time, 'time', return_value=100.01), self.assertRaises(G.OffPlanFeedback):
            G.check_air_feedback(robot, (np.zeros(6),)*4, {f'm{i}_stamp':99. for i in range(1,7)})

    def test_endpoint_tolerance_requires_final_command_and_both_position_checks(self):
        end = B.endpoint(PLAN)
        self.assertTrue(G.baseline_endpoint_reached(end, end-1, end, Q[end-1], Q))
        self.assertFalse(G.baseline_endpoint_reached(end-1, end-1, end, Q[end-1], Q))
        self.assertFalse(G.baseline_endpoint_reached(end, end-2, end, Q[end-2], Q))
        bad = Q[end-1].copy();bad[0] += np.radians(.2)
        self.assertFalse(G.baseline_endpoint_reached(end, end-1, end, bad, Q))

    def test_endpoint_index_bound_is_in_mm_and_joint_error_is_still_required(self):
        end = B.endpoint(PLAN)
        self.assertTrue(G.baseline_endpoint_reached(end, end-2, end, Q[end], Q, .25))
        self.assertFalse(G.baseline_endpoint_reached(end, end-3, end, Q[end], Q, .25))
        self.assertFalse(G.baseline_endpoint_reached(end, end-2, end, Q[end], Q, .5))
        self.assertFalse(G.baseline_endpoint_reached(end, end+1, end, Q[end], Q, .25))
        bad = Q[end].copy(); bad[2] += np.radians(.101)
        self.assertFalse(G.baseline_endpoint_reached(end, end-2, end, bad, Q, .25))
        self.assertFalse(G.baseline_endpoint_reached(end-1, end-2, end, Q[end], Q, .25))
        self.assertFalse(G.baseline_endpoint_reached(end, end-2, end, np.full(6, np.nan), Q, .25))

    def test_p2_two_index_servo_bias_completes_retracts_and_qualifies(self):
        # Fixed joint bias observed in E_E_p2_airval_02: old nearest-index gate
        # timed out despite all six joints being within 0.1 degree of the target.
        e_path = HERE/'plans/E_square100_p2_depth110.json'
        ep = json.loads(e_path.read_text()); eq = np.array(ep['q'])
        bias = np.radians([.0615584785, -.0741647306, -.0209567579,
                           .0698533775, -.0539370938, .0240595864])
        class BiasedEndpointArm(SimArm):
            def move_j(self, q):
                super().move_j(q)
                if self.k >= 439:
                    self.q = np.array(q) + bias
        with patch.dict(globals(), PLAN=ep, Q=eq, IE=np.array(ep['i_expected_A']),
                        A=np.array(ep['di_per_N_up'])):
            cal, _ = self.run_sim('p2_cal', ['--air-calibrate', '--air-clearance-mm', '150'],
                                  trials=3, plan_path=e_path)
            candidate = self.root/'p2_candidate.json'; B.build([cal], candidate)
            run, arm = self.run_sim('p2_val', ['--air-validate', '--air-clearance-mm', '150',
                                    '--air-baseline', str(candidate)], BiasedEndpointArm(seed=4), plan_path=e_path)
            tr = json.loads((run/'summary.json').read_text())['trials'][0]
            self.assertEqual(tr['status'], 'air_validated')
            self.assertEqual(tr['post_end'], 'still')
            self.assertLess(tr['t_det_s'], 11.2)
            self.assertAlmostEqual(tr['commanded_endpoint_mm']-tr['det_depth_mm'], .5)
            with np.load(run/'trial_00.npz') as d:
                self.assertTrue(np.any(d['cmd_kind'] == 2), 'backoff command required')
                self.assertFalse(np.any(d['cmd_kind'] == 5), 'must not timeout and hold')
            np.testing.assert_allclose(arm.q, eq[0])
            p = B.qualify(candidate, run, self.root/'p2_qualified.json')
            self.assertTrue(p['qualification']['passed'])
            self.assertEqual(p['qualification']['endpoint_tolerance_mm'], .5)
            # Historical policy stays strict; no retrospective success labeling.
            summary = json.loads((run/'summary.json').read_text())
            summary['trials'][0].pop('endpoint_policy')
            (run/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(B.BaselineError, 'incomplete descent'):
                B.qualify(candidate, run, self.root/'legacy_bad.json')
            summary['trials'][0]['endpoint_policy'] = B.ENDPOINT_POLICY
            summary['trials'][0]['status'] = 'timeout'
            (run/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(B.BaselineError, 'stopped early'):
                B.qualify(candidate, run, self.root/'timeout_bad.json')

    def test_qualification_requires_logged_final_command(self):
        candidate, _ = self.make_candidate()
        run, _ = self.run_sim('command_evidence', ['--air-validate', '--air-clearance-mm', '100',
                                                 '--air-baseline', str(candidate)])
        trace = run/'trial_00.npz'
        with np.load(trace) as f:
            d = {k: f[k].copy() for k in f.files}
        row = np.flatnonzero(d['state'] <= 2)[-1]
        d['cmd_val'][int(d['cmd_idx'][row])] = Q[0]
        np.savez_compressed(trace, **d)
        with self.assertRaisesRegex(B.BaselineError, 'final command'):
            B.qualify(candidate, run, self.root/'missing_command.json')

    def test_validation_one_step_servo_lag_does_not_timeout_or_extrapolate(self):
        class LaggedArm(SimArm):
            def move_j(self, q):
                requested = int(np.argmin(np.max(np.abs(Q-np.array(q)), axis=1)))
                super().move_j(Q[max(0, requested-1)])
        candidate, _ = self.make_candidate()
        run, _ = self.run_sim('lagged', ['--air-validate', '--air-clearance-mm', '100', '--air-baseline', str(candidate)], LaggedArm(seed=4))
        s = json.loads((run/'summary.json').read_text())['trials'][0]
        self.assertEqual(s['status'], 'air_validated')
        self.assertEqual(s['det_depth_mm'], (B.endpoint(PLAN)-1)*PLAN['step_mm'])
        self.assertLess(s['t_det_s'], 6)
        p = B.qualify(candidate, run, self.root/'qualified_lagged.json')
        self.assertEqual(p['qualification']['validated_end_depth_mm'], s['det_depth_mm'])
        self.assertEqual(len(p['residual_A']), B.endpoint(PLAN)+1, 'training coverage stays complete')

    def test_training_one_step_short_caps_all_future_commands_to_measured_coverage(self):
        class LaggedArm(SimArm):
            def move_j(self, q):
                requested = int(np.argmin(np.max(np.abs(Q-np.array(q)), axis=1)))
                super().move_j(Q[max(0, requested-1)])
        integer_plan = dict(PLAN, max_depth_mm=52.5)
        plan_path = self.root/'integer_plan.json';plan_path.write_text(json.dumps(integer_plan))
        cal, _ = self.run_sim('short_cal', ['--air-calibrate', '--air-clearance-mm', '100'], LaggedArm(), trials=3, plan_path=plan_path)
        candidate = self.root/'short_candidate.json';p = B.build([cal], candidate)
        self.assertEqual(p['covered_depth_mm'], 52.25)
        self.assertEqual(B.endpoint(integer_plan, p), 209)
        with self.assertRaises(B.BaselineError):
            B.expected_residual(p, 210)
        run, _ = self.run_sim('short_val', ['--air-validate', '--air-clearance-mm', '100', '--air-baseline', str(candidate)], SimArm(seed=5), plan_path=plan_path)
        data = np.load(run/'trial_00.npz')
        commanded = [np.argmin(np.max(np.abs(Q-q), axis=1)) for q in data['cmd_val']]
        self.assertLessEqual(max(commanded), 209)
        qualified = B.qualify(candidate, run, self.root/'short_qualified.json')
        self.assertEqual(qualified['qualification']['command_limit_mm'], 52.25)

    def test_profile_more_than_one_step_short_is_rejected(self):
        with self.assertRaises(B.BaselineError):
            B.endpoint(PLAN, {'residual_A': np.zeros((B.endpoint(PLAN)-1, 6)).tolist()})


if __name__ == '__main__':
    unittest.main()

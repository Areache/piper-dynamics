"""Payload estimation, checked by synthetic round-trip.

No URDF and no hardware needed: build a regressor, synthesise torques from a
known mass, check the mass comes back. This is the test that would catch a
sign flip or a normalisation error in the fit.
"""

import numpy as np

from piper_dynamics.estimation.payload import estimate_payload_mass_from_lsq


def test_recovers_known_mass_exactly_without_noise():
    rng = np.random.default_rng(0)
    b = rng.normal(0, 1, (200, 6))
    m_true = 0.734
    tau = m_true * b
    est = estimate_payload_mass_from_lsq(b, tau)
    np.testing.assert_allclose(est.mass_kg, m_true, rtol=1e-9)
    assert est.stderr_kg < 1e-9


def test_noise_shows_up_in_stderr_not_bias():
    rng = np.random.default_rng(1)
    b = rng.normal(0, 1, (4000, 6))
    m_true = 1.25
    tau = m_true * b + rng.normal(0, 0.05, b.shape)
    est = estimate_payload_mass_from_lsq(b, tau)
    assert abs(est.mass_kg - m_true) < 5 * est.stderr_kg
    assert est.stderr_kg > 0


def test_unobservable_window_is_reported_not_guessed():
    b = np.zeros((50, 6))
    est = estimate_payload_mass_from_lsq(b, np.zeros((50, 6)))
    assert np.isnan(est.mass_kg)
    assert not est.observable

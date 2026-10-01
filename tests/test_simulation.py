import sys
sys.path.insert(0, "src")

import numpy as np
import pytest

from option_hedging.simulation import sigma_bar, simulate_gbm_paths

def test_sigma_bar_matches_sigma_t_T_at_maturity():
    # sigma_bar(t, T, T) should match the closed-form sigma_t_T 
    T = 1.40
    for t in [0, 0.25, 0.5, 0.75, 1.0, 1.3]:
        f = 0.04382 - 0.0169*t - 0.00835714*t**2 - 0.00137755*t**3
        expected = np.sqrt(f / (T - t))
        assert sigma_bar(t, T, T) == pytest.approx(expected, abs=1e-6)

def test_simulate_gbm_paths_starts_at_S0():
    S, times = simulate_gbm_paths(107, 0.14, 1.40, 32, 1000)
    assert np.allclose(S[:, 0], 107)

def test_simulate_gbm_paths_shape():
    S, times = simulate_gbm_paths(107, 0.14, 1.40, 32, 1000)
    assert S.shape == (1000, 33)
    assert times.shape == (33,)
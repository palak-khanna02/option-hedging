import sys
sys.path.insert(0, "src")

import numpy as np
import pytest

from option_hedging.simulation import simulate_gbm_paths
from option_hedging.pricing import black_scholes_call_price, black_scholes_put_price, V_t_func

S0, mu, T, r = 107.0, 0.14, 1.40, 0.04
K_c, K_p = 90, 107


def test_option_price_at_t0():
    V0 = black_scholes_call_price(S0, 0, K_c, T, r, None) + black_scholes_put_price(S0, 0, K_p, T, r, None)
    assert V0 == pytest.approx(29.3537271, abs=1e-4)

def test_maturity_column_is_payoff_not_zero():
    # regression test: an earlier version's pricing loop stopped one step
    # short of maturity, leaving the last column at its default zero, so
    # every strategy holding options was valued as if the option paid out
    # nothing at expiry. 
    S, times = simulate_gbm_paths(S0, mu, T, 32, 2000)
    V = V_t_func(S0, S, times, K_c, K_p, T, r, None, 2000)

    S_T = S[:, -1]
    expected_payoff = np.maximum(S_T - K_c, 0) + np.maximum(-S_T + K_p, 0)

    assert not np.allclose(V[:, -1], 0.0)
    assert np.allclose(V[:, -1], expected_payoff)

def test_first_column_constant_across_paths():
    # all paths start from the same S0, so t=0 price should be identical everywhere
    S, times = simulate_gbm_paths(S0, mu, T, 32, 500)
    V = V_t_func(S0, S, times, K_c, K_p, T, r, None, 500)
    assert np.allclose(V[:, 0], V[0, 0])
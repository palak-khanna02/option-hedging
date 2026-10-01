import sys
sys.path.insert(0, "src")

import numpy as np

from option_hedging.simulation import simulate_gbm_paths
from option_hedging.hedging import simulate_delta_hedging

S0, mu, T, r = 107.0, 0.14, 1.40, 0.04
K_c, K_p = 90, 107
b0, theta = 132, 0.01


def test_do_nothing_strategy_is_deterministic():
    # q=(0,0,0) trades nothing and holds no risky asset, so every path
    # should give the SAME final wealth, exactly b0*exp(r*T) -- no
    # randomness possible, since nothing about the strategy depends on S.
    S, times = simulate_gbm_paths(S0, mu, T, 32, 2000)
    wealth = simulate_delta_hedging(S0, K_c, K_p, T, r, mu, None, 32, 2000, theta, (0,0,0), b0, S, times)
    expected = b0 * np.exp(r * T)
    assert np.allclose(wealth, expected)


def test_zero_theta_gives_higher_wealth_than_with_fees():
    # removing transaction costs should never make a risky strategy worse off
    S, times = simulate_gbm_paths(S0, mu, T, 32, 2000)
    q = (1, 1, 1)
    wealth_with_fees = simulate_delta_hedging(S0, K_c, K_p, T, r, mu, None, 32, 2000, 0.01, q, b0, S, times)
    wealth_no_fees = simulate_delta_hedging(S0, K_c, K_p, T, r, mu, None, 32, 2000, 0.0, q, b0, S, times)
    assert wealth_no_fees.mean() > wealth_with_fees.mean()
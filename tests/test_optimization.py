import sys
sys.path.insert(0, "src")

import numpy as np
import pytest

from option_hedging.optimization import phi
from option_hedging.optimization import generate_qopt

S0, mu, T, r = 107.0, 0.14, 1.40, 0.04
b0, lbda = 132, 0.01
K_c, K_p = 90, 107
theta = 0.01


def test_phi_of_deterministic_wealth_equals_exp_rT():
    # for zero-variance wealth b0*exp(r*T), phi collapses to exp(r*T)
    # exactly, independent of lambda -- good check that the phi formula
    # (the -1/(b0*lambda)*log(...) scaling) is implemented correctly.
    wealth = np.full(10000, b0 * np.exp(r * T))
    result = phi(wealth, lbda, b0)
    assert result == pytest.approx(np.exp(r * T), abs=1e-6)

def test_optimizer_does_not_return_unaffordable_position():
    # regression test: an earlier version of the budget constraint had a
    # sign error that let the optimizer run away to positions worth
    # hundreds of millions against a starting budget of just 132. A
    # correctly-constrained optimum should stay within a sane multiple of
    # b0 / typical asset prices, not blow up to that scale.
    qOpt, res = generate_qopt(S0, K_c, K_p, T, r, mu, theta, lbda, b0, n_steps=32, n_paths=2000)
    assert np.all(np.abs(qOpt) < 100), f"qOpt blew up unexpectedly: {qOpt}"
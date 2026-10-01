"""
option_hedging
==============

A small library for simulating a strangle-like option (long call K=90,
long put K=107) under a time-varying-volatility GBM stock, delta-hedging
it under transaction costs, and finding the utility-maximising static
trading rule (q1, q2, q3) via constrained optimisation.

Modules
-------
simulation    : stock-price simulation (exact scheme via time-averaged volatility)
pricing       : Black-Scholes pricing/greeks using the time-averaged volatility
hedging       : wealth simulation for a given (q1, q2, q3) strategy
optimization  : constrained utility optimisation over (q1, q2, q3)
"""

from .simulation import sigma, sigma_bar, simulate_gbm_paths
from .pricing import (
    N,
    sigma_t_T,
    compute_d1_and_d2,
    black_scholes_call_price,
    black_scholes_put_price,
    black_scholes_call_delta,
    black_scholes_put_delta,
    V_t_func,
)
from .hedging import simulate_delta_hedging
from .optimization import phi, generate_qopt

__all__ = [
    "sigma",
    "sigma_bar",
    "simulate_gbm_paths",
    "N",
    "sigma_t_T",
    "compute_d1_and_d2",
    "black_scholes_call_price",
    "black_scholes_put_price",
    "black_scholes_call_delta",
    "black_scholes_put_delta",
    "V_t_func",
    "simulate_delta_hedging",
    "phi",
    "generate_qopt",
]
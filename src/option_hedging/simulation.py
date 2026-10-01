"""
Stock price simulation.

The model's instantaneous volatility is

    sigma(t) = 0.13 + 0.09 * (t / T)

Because sigma(t) is deterministic (not stochastic), the stochastic integral
int_t^s sigma(u) dW_u is *exactly* Normal with mean 0 and variance
int_t^s sigma(u)^2 du, for any step size s - t. So simulating log(S) with
the *interval-averaged* volatility

    sigma_bar(t, s) = sqrt( (1 / (s - t)) * int_t^s sigma(u)^2 du )

in place of the pointwise sigma(t) at each step gives an *exact* simulation
of the SDE, with no discretisation error, however large the step. Using the
pointwise sigma(t) instead (as a naive discretisation would) introduces a
small but real bias -- see tests/test_simulation.py.
"""

import numpy as np


def sigma(t, T):
    return 0.13 + 0.09 * (t / T) # Instantaneous volatility at time t, for an option maturing at T.

# average vol over [t,s], used to simulate exact solution to SDE
def sigma_bar(t,s,T):
    integral= 0.0169*(s-t)+(0.0117/T)*(s**2-t**2) + (0.0027/T**2)*(s**3-t**3)
    return np.sqrt(integral/(s-t))

def simulate_gbm_paths(S0, mu,T,n_steps, n_paths):
    Z = np.zeros((n_paths, n_steps + 1))
    Z[:, 0] = np.log(S0)
    dt=T/n_steps
    times = np.linspace(0,T,n_steps+1)
    epsilon = np.random.randn(n_paths, n_steps)
    for i in range(0,n_steps):
        t,s = times[i], times[i+1]
        sbar = sigma_bar(t,s,T)
        Z[:,i+1] = Z[:,i] + (mu -0.5*sbar**2) *dt +sbar*np.sqrt(dt)*epsilon[:,i]
    
    S=np.exp(Z)
    return S, times
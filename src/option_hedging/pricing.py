import numpy as np
import scipy.stats
from scipy.stats import norm

from .simulation import sigma_bar


# standard normal CDF, used in the Black-Scholes formulas below
def N(x):
    return scipy.stats.norm.cdf(x)

# average volatility from t to maturity T, used to price the option at any time t < T
def sigma_t_T(T, t):
    f = 0.04382 - 0.0169*t - 0.00835714*(t**2) - 0.00137755*(t**3)
    sigma = np.sqrt((1/(T-t))*f)
    return sigma

# standard Black-Scholes d1, d2 terms
def compute_d1_and_d2(S, t, K, T, r, sigma):
    tau = T-t
    d1 = 1/(sigma*np.sqrt(tau))*(np.log(S/K) + (r+0.5*sigma**2)*tau)
    d2 = d1 - sigma*np.sqrt(tau)
    return d1, d2

# price of a call option, using sigma_t_T as the volatility
def black_scholes_call_price(S, t, K, T, r, sigma):
    sigma = sigma_t_T(T, t)
    d1, d2 = compute_d1_and_d2(S, t, K, T, r, sigma)
    return S*N(d1) - np.exp(-r*(T-t))*K*N(d2)

# price of a put option
def black_scholes_put_price(S, t, K, T, r, sigma):
    sigma = sigma_t_T(T, t)
    d1, d2 = compute_d1_and_d2(S, t, K, T, r, sigma)
    return -S*N(-d1) + np.exp(-r*(T-t))*K*N(-d2)

# call option's delta (sensitivity to S)
def black_scholes_call_delta(S, t, K, T, r, sigma):
    sigma = sigma_t_T(T, t)
    d1, d2 = compute_d1_and_d2(S, t, K, T, r, sigma)
    return N(d1)

# put option's delta
def black_scholes_put_delta(S, t, K, T, r, sigma):
    sigma = sigma_t_T(T, t)
    d1, d2 = compute_d1_and_d2(S, t, K, T, r, sigma)
    return N(d1)-1

# prices the combined call+put position at every (path, time) point
def V_t_func(S0, S, times, K_c, K_p, T, r, sigma, n_paths):
    BS_price = np.zeros(S.shape)
    BS_price[:,0] = black_scholes_call_price(S0,0,K_c,T,r,sigma_t_T)+black_scholes_put_price(S0,0,K_p,T,r,sigma_t_T)
    for i in range(1, len(times)-1):
        t = times[i]
        S_t = S[:,i]
        BS_price[:,i] = black_scholes_call_price(S_t,t,K_c,T,r,sigma_t_T)+black_scholes_put_price(S_t,t,K_p,T,r,sigma_t_T)
    BS_price[:,-1] = np.maximum(S[:,-1]-K_c,0) + np.maximum(-S[:,-1]+K_p,0)
    return BS_price
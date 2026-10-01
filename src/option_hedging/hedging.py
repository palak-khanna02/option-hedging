import numpy as np

from .pricing import black_scholes_call_delta, black_scholes_put_delta, V_t_func, sigma_t_T


def simulate_delta_hedging(S0, K_c, K_p, T, r, mu, sigma, n_steps, n_paths, theta, q, b0, S, times):
    V = V_t_func(S0, S, times, K_c, K_p, T, r, sigma_t_T, n_paths)
    Pi = np.zeros(S.shape)
    Pi[:,0] = b0
    dt = T/n_steps

    prev_stock_qty = np.zeros(n_paths)
    prev_option_qty = 0

    for i in range(0, len(times)-1):
        t = times[i]
        S_t = S[:,i]
        V_t = V[:,i]
        qt_call = black_scholes_call_delta(S_t,t,K_c,T,r,sigma_t_T)
        qt_put = black_scholes_put_delta(S_t,t,K_p,T,r,sigma_t_T)
        q_t = qt_call - abs(qt_put)

        curr_stock_qty = q[0] + q[1]*q_t
        curr_option_qty = q[2]

        # changed — fee on the CHANGE in position (what's traded), not the full holding
        stock_traded = curr_stock_qty - prev_stock_qty
        option_traded = curr_option_qty - prev_option_qty

        B_t = Pi[:,i] - q[2]*V_t - curr_stock_qty*S_t \
              - abs(stock_traded)*theta*S_t \
              - abs(option_traded)*theta*V_t

        B_with_interest = B_t*np.exp(dt*r)
        S_tplusdt = S[:,i+1]
        new_stock_value = curr_stock_qty*S_tplusdt
        V_tplusdt = V[:,i+1]
        Pi[:,i+1] = B_with_interest + new_stock_value + q[2]*V_tplusdt

        prev_stock_qty = curr_stock_qty
        prev_option_qty = curr_option_qty

    wealth = Pi[:,-1]
    return wealth
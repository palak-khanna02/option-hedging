import numpy as np
import scipy.optimize
import scipy.special
from tqdm import tqdm

from .simulation import simulate_gbm_paths
from .pricing import black_scholes_call_price, black_scholes_put_price, black_scholes_call_delta, black_scholes_put_delta
from .hedging import simulate_delta_hedging

# scaled utility measure: larger phi = better. phi(0,0,0) should equal exp(r*T) exactly.
def phi(wealth, lbda, b0):
    n = len(wealth)
    log_mean_neg_u = scipy.special.logsumexp(-lbda*wealth) - np.log(n)
    return -log_mean_neg_u/(b0*lbda)

def generate_qopt(S0, K_c, K_p, T, r, mu, theta, lbda, b0, n_steps, n_paths, x0=None):
    S, times = simulate_gbm_paths(S0, mu, T, n_steps, n_paths)

    option_prices = black_scholes_call_price(S0,0,K_c,T,r,None) + black_scholes_put_price(S0,0,K_p,T,r,None)
    q_t0 = black_scholes_call_delta(S0,0,K_c,T,r,None) - abs(black_scholes_put_delta(S0,0,K_p,T,r,None))

    def constraint_function(q):
        # changed — fixed sign bug: every cost term now subtracted, not added
        return b0 - q[2]*option_prices - (q[0]+q[1]*q_t0)*S0 - abs(q[0]+q[1]*q_t0)*theta*S0 - abs(q[2])*theta*option_prices

    pbar = tqdm(desc="Objective evaluations", unit="eval")

    def objective(q):
        # changed — now directly equals -phi(q), so res.fun can be read as -phi_optimal
        final_balance = simulate_delta_hedging(S0, K_c, K_p, T, r, mu, None, n_steps, n_paths, theta, q, b0, S, times)
        pbar.update(1)
        n = len(final_balance)
        log_mean_neg_u = scipy.special.logsumexp(-lbda*final_balance) - np.log(n)
        return log_mean_neg_u/(b0*lbda)

    constraintDic = {"type": "ineq", "fun": constraint_function}
    start = x0 if x0 is not None else np.ones(3)

    res = scipy.optimize.minimize(objective,
                                    start,
                                    constraints=[constraintDic],
                                    method='SLSQP')
    pbar.close()
    assert res['success']
    qOpt = res['x']
    return qOpt, res
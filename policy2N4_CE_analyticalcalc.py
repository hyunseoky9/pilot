"""
Analytical CE calculation for policy types 2 and 4 with LOGNORMAL returns.
Both have deterministic allocation paths (no belief updating).
Uses 3D Gauss-Hermite quadrature (one dimension per region) since
the portfolio return is a sum of independent lognormals.
"""
import numpy as np
from scipy.special import logsumexp
from numpy.polynomial.hermite import hermgauss
from pprdyn1 import pprdyn1

# Gauss-Hermite quadrature points and weights
# Use fewer points per dimension since we're doing 3D: 20^3 = 8000 points
N_QUAD_1D = 20
XI_1D, WI_1D = hermgauss(N_QUAD_1D)

# Build 3D tensor product grid
XI_3D = np.array(np.meshgrid(XI_1D, XI_1D, XI_1D)).T.reshape(-1, 3)  # (8000, 3)
WI_3D = np.outer(np.outer(WI_1D, WI_1D).ravel(), WI_1D).ravel()      # (8000,)
LOG_WI_3D = np.log(WI_3D)
LOG_NORM_3D = -1.5 * np.log(np.pi)  # normalization: 1/pi^{3/2}


def analytical_ce(env, policytype):
    assert policytype in [2, 4], "Only policies 2 and 4 are deterministic."
    
    gamma = env.gamma
    sigma = env.sig
    beta = env.beta
    T = env.T
    n_scenario = env.n_scenario
    n_region = env.n_region
    probs = np.ones(n_scenario) / n_scenario
    
    best_wmat = env.best_weight_idx_matrix
    uniform_bidx = env.b_states.shape[0] - 1
    
    # Step 1: Build deterministic allocation path
    A_path = np.zeros((T + 1, n_region))
    w_path = np.zeros((T, n_region))
    
    for t in range(T):
        if policytype == 2:
            actionidx = best_wmat[uniform_bidx, -1]
        elif policytype == 4:
            actionidx = best_wmat[uniform_bidx, t + 1]
        
        w = env.w_states[actionidx]
        w_path[t] = w
        
        if t == 0:
            A_next = w.copy()
        else:
            A_next = t / (t + 1) * A_path[t] + 1 / (t + 1) * w
        
        dists = np.linalg.norm(env.A_states - A_next, axis=1)
        A_next = env.A_states[np.argmin(dists)]
        A_path[t + 1] = A_next
    
    print(f"Policy {policytype} allocation path:")
    for t in range(T):
        print(f"  t={t}: action={w_path[t]}, A_{t+1}={A_path[t+1]}")
    
    # Step 2 & 3: Compute per-scenario discounted PV, then average
    K = np.sum(beta ** np.arange(T))
    logK = np.log(K)
    log_betas = np.arange(T) * np.log(beta)
    
    def compute_lognormal_returns(mu_s):
        """
        Compute lognormal return samples at all quadrature points.
        mu_s: (3,) mean returns per region under scenario s
        Returns: (8000, 3) array of return values at each quadrature point
        
        Each region i: r_i = exp(log(mu_i) - 0.5*sigma^2 + sigma*sqrt(2)*xi)
        so that E[r_i] = mu_i
        """
        log_mu = np.log(np.maximum(mu_s, 1e-10))
        log_r = log_mu[np.newaxis, :] - 0.5 * sigma**2 + sigma * np.sqrt(2) * XI_3D
        return np.exp(log_r)
    
    def compute_log_expected_neg_utility(A_t, scale_factor, mu_s):
        """
        Compute log E[-u(R_p)] where R_p = scale_factor * A_t . r
        and r_i ~ Lognormal with E[r_i] = mu_s[i], independently.
        For gamma > 1.
        """
        r_points = compute_lognormal_returns(mu_s)  # (8000, 3)
        port_returns = scale_factor * (r_points @ A_t)  # (8000,)
        port_returns = np.maximum(port_returns, 1e-300)
        
        log_neg_u = (1 - gamma) * np.log(port_returns) - np.log(gamma - 1)
        log_terms = LOG_WI_3D + log_neg_u
        return logsumexp(log_terms) + LOG_NORM_3D
    
    def compute_expected_log_utility(A_t, scale_factor, mu_s):
        """Compute E[log(R_p)] for gamma = 1."""
        r_points = compute_lognormal_returns(mu_s)
        port_returns = scale_factor * (r_points @ A_t)
        port_returns = np.maximum(port_returns, 1e-300)
        log_vals = np.log(port_returns)
        return np.sum(WI_3D * log_vals) * np.pi**(-1.5)
    
    def compute_expected_utility_low_gamma(A_t, scale_factor, mu_s):
        """Compute E[u(R_p)] for gamma < 1."""
        r_points = compute_lognormal_returns(mu_s)
        port_returns = scale_factor * (r_points @ A_t)
        port_returns = np.maximum(port_returns, 1e-300)
        u_vals = port_returns ** (1 - gamma) / (1 - gamma)
        return np.sum(WI_3D * u_vals) * np.pi**(-1.5)
    
    if gamma > 1:
        log_neg_PV_per_scenario = np.zeros(n_scenario)
        
        for s in range(n_scenario):
            log_neg_u_per_period = np.zeros(T)
            for t in range(T):
                A_t = A_path[t + 1]
                scale_factor = t + 1
                mu_s = env.multi_timestep_returns[t + 1, :, s]
                
                val = compute_log_expected_neg_utility(A_t, scale_factor, mu_s)
                log_neg_u_per_period[t] = val
            print(f"  Scenario {s}: log_neg_u per period = {np.round(log_neg_u_per_period, 4)}")
            log_neg_PV_per_scenario[s] = logsumexp(log_neg_u_per_period + log_betas)
            print(f"  Scenario {s}: log(-PV|s) = {log_neg_PV_per_scenario[s]:.6f}")
        
        log_probs = np.log(probs)
        log_neg_PV = logsumexp(log_neg_PV_per_scenario + log_probs)
        
        logA = np.log(gamma - 1) + log_neg_PV - logK
        log_ce = logA / (1 - gamma)
        ce = np.exp(log_ce)
        
        print(f"\nlog(-PV) = {log_neg_PV:.6f}")
        print(f"log(CE) = {log_ce:.6f}")
        print(f"CE = {ce:.6f}")
    
    elif np.isclose(gamma, 1.0):
        PV_per_scenario = np.zeros(n_scenario)
        for s in range(n_scenario):
            for t in range(T):
                A_t = A_path[t + 1]
                scale_factor = t + 1
                mu_s = env.multi_timestep_returns[t + 1, :, s]
                
                val = compute_expected_log_utility(A_t, scale_factor, mu_s)
                PV_per_scenario[s] += beta ** t * val
        
        V = np.dot(probs, PV_per_scenario)
        ce = np.exp(V / K)
        print(f"\nV = {V:.6f}")
        print(f"CE = {ce:.6f}")
    
    else:  # gamma < 1
        PV_per_scenario = np.zeros(n_scenario)
        for s in range(n_scenario):
            for t in range(T):
                A_t = A_path[t + 1]
                scale_factor = t + 1
                mu_s = env.multi_timestep_returns[t + 1, :, s]
                
                val = compute_expected_utility_low_gamma(A_t, scale_factor, mu_s)
                PV_per_scenario[s] += beta ** t * val
        
        V = np.dot(probs, PV_per_scenario)
        ce = ((1 - gamma) * V / K) ** (1 / (1 - gamma))
        print(f"\nV = {V:.6f}")
        print(f"CE = {ce:.6f}")
    
    # After building A_path, test one period:
    t = 0
    s = 0
    A_t = A_path[1]
    mu_s = env.multi_timestep_returns[1, :, s]
    scale_factor = 1

    # Analytical
    val_analytical = compute_log_expected_neg_utility(A_t, scale_factor, mu_s)

    # MC for same period and scenario
    N_mc = 100000
    log_returns = np.log(mu_s) - 0.5*sigma**2 + sigma*np.random.randn(N_mc, 3)
    r_samples = np.exp(log_returns)
    port_returns = scale_factor * (r_samples @ A_t)
    neg_u = port_returns**(1-gamma) / (gamma-1)
    val_mc = np.log(np.mean(neg_u))

    print(f"log E[-u] analytical: {val_analytical:.6f}")
    print(f"log E[-u] MC:         {val_mc:.6f}")

    return ce


if __name__ == "__main__":
    settings = {'settingID': 4}

    env = pprdyn1(settings)
    
    print("=" * 60)
    print("POLICY 2: Static final-period optimization")
    print("=" * 60)
    ce2 = analytical_ce(env, policytype=2)
    
    print("\n" + "=" * 60)
    print("POLICY 4: Rolling next-period optimization (fixed beliefs)")
    print("=" * 60)
    ce4 = analytical_ce(env, policytype=4)
    
    

    print(f"\n{'=' * 60}")
    print(f"Policy 2 CE: {ce2:.6f}")
    print(f"Policy 4 CE: {ce4:.6f}")
    print(f"Welfare gain of 4 over 2: {(ce4 - ce2) / ce2 * 100:.2f}%")

import numpy as np
from scipy.optimize import minimize

class utilitysolver:
    def __init__(self, risk_aversion):
        self.risk_aversion = risk_aversion

    def _crra_utility(self, r):
        gamma = self.risk_aversion
        r = np.asarray(r, dtype=float)
        if np.any(r <= 0):
            raise ValueError("CRRA utility is undefined for non-positive returns.")
        if np.isclose(gamma, 1.0):
            return np.log(r)
        else:
            return r ** (1.0 - gamma) / (1.0 - gamma)

    def max_utility(self, returns, probs=None, short_selling=False):
        '''
        Get portfolio weights that maximize expected CRRA utility
        using log-sum-exp formulation with analytical gradients.

        Parameters
        ----------
        returns : ndarray, shape (n_assets, n_scenarios)
        probs : ndarray or None, shape (n_scenarios,)
        short_selling : bool

        Returns
        -------
        max_utility : float  (log-transformed objective, not true utility)
        weights : ndarray
        '''
        returns = np.asarray(returns, dtype=float)
        if returns.ndim != 2:
            raise ValueError("returns must have shape (n_assets, n_scenarios).")

        n_assets, n_scenarios = returns.shape

        if probs is None:
            probs = np.ones(n_scenarios) / n_scenarios
        else:
            probs = np.asarray(probs, dtype=float)
            if probs.shape != (n_scenarios,):
                raise ValueError("probs must have shape (n_scenarios,).")
            if np.any(probs < 0):
                raise ValueError("Probabilities must be nonnegative.")
            probs = probs / probs.sum()

        gamma = self.risk_aversion

        def objective_and_grad(w):
            port_returns = w @ returns  # (n_scenarios,)

            if np.any(port_returns <= 0):
                return 1e12, np.zeros(n_assets)

            if np.isclose(gamma, 1.0):
                log_port = np.log(port_returns)
                val = np.sum(probs * log_port)
                grad = returns @ (probs / port_returns)
                return -val, -grad
            else:
                log_port = np.log(port_returns)
                exponents = (1.0 - gamma) * log_port
                max_exp = np.max(exponents)
                shifted = np.exp(exponents - max_exp)
                denom = np.sum(probs * shifted)
                log_obj = max_exp + np.log(denom)

                scenario_weights = (probs * shifted) / denom
                grad_log = (1.0 - gamma) * returns @ (scenario_weights / port_returns)

                if gamma < 1.0:
                    return -log_obj, -grad_log
                else:
                    return log_obj, grad_log

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        if short_selling:
            bounds = None
        else:
            bounds = [(0.0, 1.0) for _ in range(n_assets)]

        init_guess = np.ones(n_assets) / n_assets

        res = minimize(
            objective_and_grad,
            init_guess,
            method='SLSQP',
            jac=True,
            bounds=bounds,
            constraints=constraints
        )

        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        return -res.fun, res.x

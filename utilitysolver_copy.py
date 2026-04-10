import numpy as np
from scipy.optimize import minimize

class utilitysolver:
    def __init__(self, risk_aversion):
        self.risk_aversion = risk_aversion

    def _crra_utility(self, r):
        """
        CRRA utility applied directly to returns.

        u(r) = log(r),                  if gamma = 1
             = r^(1-gamma)/(1-gamma),  if gamma != 1

        Requires r > 0.
        """
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
        when utility is a function of portfolio returns directly.

        Parameters
        ----------
        returns : ndarray
            Shape (n_assets, n_scenarios)
        probs : ndarray or None
            Scenario probabilities, shape (n_scenarios,)
            If None, equal probabilities are assumed.
        short_selling : bool
            If False, enforce 0 <= w_i <= 1 and sum(w)=1.
            If True, only enforce sum(w)=1.

        Returns
        -------
        max_utility : float
            Maximum expected utility
        weights : ndarray
            Optimal portfolio weights
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

        def objective(w):
            port_returns = w @ returns

            if np.any(port_returns <= 0):
                return 1e12

            if np.isclose(gamma, 1.0):
                # E[log(r)] — no numerical issues, maximize directly
                return -np.sum(probs * np.log(port_returns))
            else:
                # Log-sum-exp formulation for numerical stability
                log_port = np.log(port_returns)
                exponents = (1.0 - gamma) * log_port
                max_exp = np.max(exponents)
                log_obj = max_exp + np.log(np.sum(probs * np.exp(exponents - max_exp)))

                if gamma < 1.0:
                    # u = r^(1-γ)/(1-γ), (1-γ)>0, maximize => minimize negative
                    return -log_obj
                else:
                    # u = r^(1-γ)/(1-γ), (1-γ)<0, E[u] < 0
                    # maximize E[u] => minimize E[r^(1-γ)] => minimize log_obj
                    return log_obj

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        if short_selling:
            bounds = None
        else:
            bounds = [(0.0, 1.0) for _ in range(n_assets)]

        init_guess = np.ones(n_assets) / n_assets

        res = minimize(
            objective,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        return -res.fun, res.x

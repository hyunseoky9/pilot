import numpy as np
from scipy.optimize import minimize

class utilitysolver:
    def __init__(self, risk_aversion):
        self.risk_aversion = risk_aversion

    def max_utility(self, returns, probs=None, short_selling=False):
        '''
        Get portfolio weights that maximize mean-variance utility:

            U = E[r_p] - risk_aversion * Var(r_p)

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
            Maximum mean-variance utility
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

        expected_returns = returns @ probs

        centered = returns - expected_returns[:, None]
        cov_matrix = centered @ np.diag(probs) @ centered.T

        def objective(w):
            mean_return = w @ expected_returns
            variance = w @ cov_matrix @ w
            utility = mean_return - self.risk_aversion * variance
            return -utility

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

        optimal_weights = res.x
        max_utility = -res.fun

        return max_utility, optimal_weights
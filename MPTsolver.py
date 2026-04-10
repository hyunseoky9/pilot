import numpy as np
from scipy.optimize import minimize

class MPTsolver:
    def __init__(self, returns):
        self.returns = returns
        self.expected_returns = np.mean(returns, axis=1)
        self.cov_matrix = np.cov(returns, ddof=0)

    def get_efficient_frontier(self, numportfolios=100):
        min_ret = np.min(self.expected_returns)
        max_ret = np.max(self.expected_returns)
        target_returns = np.linspace(min_ret, max_ret, numportfolios)

        frontier_points = []
        weights_record = []

        n_assets = len(self.expected_returns)
        # Initial guess: equal distribution
        init_guess = np.ones(n_assets) / n_assets
        # Constraint: Weights must be >= 0 (No short-selling)
        bounds = tuple((0, 1) for _ in range(n_assets))

        for target in target_returns:
            # Constraints: 1) Sum of weights = 1, 2) Expected return = target
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, self.expected_returns) - target}
            ]
            
            # Objective function: Minimize Portfolio Variance
            def port_variance(w):
                return np.dot(w.T, np.dot(self.cov_matrix, w))

            res = minimize(port_variance, init_guess, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
            
            if res.success:
                weights = res.x
                frontier_points.append([target, np.sqrt(res.fun)])
                weights_record.append(weights)

        frontier = np.array(frontier_points)
        if frontier.size == 0:
            raise ValueError("No feasible portfolios found for the provided returns.")

        # Keep only the upper (efficient) branch from the global minimum-variance point.
        min_risk_idx = np.argmin(frontier[:, 1])
        frontier = frontier[min_risk_idx:]
        weights_record = weights_record[min_risk_idx:]

        return frontier, weights_record

        
        
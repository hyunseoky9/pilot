from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp
from scipy.stats import norm
import numpy as np
import pickle
import pandas as pd
from itertools import product
class pprdyn1:
    def __init__(self,settings):
        '''
        dynamic version of the PPR model from Mallory and Ando (2014), check out PPR dynamic model for details.

        '''
        self.envID = 'pprdyn1'
        envsetting = settings['settingID']
        self.settingID = envsetting

        # parameters
        # import csv of parameters
        filename = 'envsetting.csv'
        params = pd.read_csv(filename)
        self.T = params['T'].iloc[envsetting]
        self.sig = params['sig'].iloc[envsetting]
        self.beta = params['beta'].iloc[envsetting]
        self.gamma = params['gamma'].iloc[envsetting]
        self.central_adjustment = params['centraladj'].iloc[envsetting]
        self.bnorm = params['bnorm'].iloc[envsetting] # budget normalization
        self.n_region = 3
        self.n_scenario = 4

        # data from Mallory and Ando (2014) Table 2 baseline land value scenario
        self.returnsdata = np.array([
        [2.07, 1.49, 1.21, 1.44],  # Western
        [3.70, 3.11, 1.61, 2.74],  # Central
        [1.10, 1.72, 1.78, 1.96],  # Eastern
        ])  # col 0: historical, col 1-3: future scenarios
        self.returnsdata[1] = self.returnsdata[1] - self.central_adjustment # altering central region values to make it more comparable

        # reward version (0=for general evaluation, 1=for training RL with actual crra utility)
        self.for_RL = settings.get('for_RL', 0)
        if self.for_RL == 1:
            self.absref_U = np.abs((settings['ref_R']**(1 - self.gamma))/(1 - self.gamma)) # reference return to divide reward with to keep the numerical scale of rewards more manageable for RL training
            self.R_clip = np.min(self.returnsdata) * np.exp(-0.5*self.sig**2 - 5.4*self.sig) # 5.4 = amount of sd considered in the quadrature approximation. 0.4 lowest mean return 

        # get linearly interpolated future return means
        self.multi_timestep_returns = np.zeros((self.T+1,) + self.returnsdata.shape)
        self.multi_timestep_returns[-1] = self.returnsdata

        self.multi_timestep_returns[0] = np.tile(self.returnsdata[:, 0], (4, 1)).T
        historical_col = self.returnsdata[:, 0][:, np.newaxis]
        for t in range(1, self.T):
            self.multi_timestep_returns[t] = historical_col + (self.returnsdata - historical_col) * t / (self.T)

        # discretize state variables
        self.A_dim = (self.n_region)
        self.b_dim = (self.n_scenario)
        self.t_dim = (1)
        self.s_dim = (1)
        self.statevar_dim = (self.A_dim, self.b_dim, self.s_dim, self.t_dim)
        self.sidx =  {}
        varnames = ['A', 'b', 's', 't']
        idxaccum = 0
        for  i, name in enumerate(varnames):
            self.sidx[name] = np.arange(self.statevar_dim[i]).astype(int) + idxaccum
            idxaccum += self.statevar_dim[i]

        # for oidx, just take out 's'
        self.oidx = {k:v for k,v in self.sidx.items() if k != 's'}
        self.aidx = {'w_w':0,'w_c':1,'w_e':2}
        # discretize action and state variables (A, b, and w)
        ## make 5% increments of A that sum to 100%
        increments = np.arange(0, 21)  # 0%, 5%, ..., 100%
        self.A_states = np.array(
            [combo for combo in product(increments, repeat=self.n_region) if sum(combo) == 20],
            dtype=int,
        )
        self.A_states = self.A_states*5 / 100.0 # convert from integer increments to fractions summing to 1
        self.n_A_states = self.A_states.shape[0]
        ## make 10% increments of b that sum to 100%
        increments = np.arange(0, 11)  # 0%, 10%, ..., 100%
        self.b_states = np.array(
            [combo for combo in product(increments, repeat=self.n_scenario) if sum(combo) == 10],
            dtype=int,
        )
        self.b_states = self.b_states / 10.0
        # add equal allocation belief state
        self.b_states = np.vstack([self.b_states, np.ones(self.n_scenario)/self.n_scenario])
        # Enforce a strictly positive floor so no belief component is ever exactly zero.
        self.belief_epsilon = float(settings.get('belief_epsilon', 1e-6))
        self.b_states = np.maximum(self.b_states, self.belief_epsilon)
        self.b_states = self.b_states / self.b_states.sum(axis=1, keepdims=True)
        
        self.n_b_states = self.b_states.shape[0]

        ## action (allocation weight of budget across regions)
        self.actionvar_dim = (1,1,1)
        self.w_states = self.A_states
        self.n_w_states = self.w_states.shape[0]

        # state, obs, and action space dimensions
        self.obsspace_dim = self.n_region + self.n_scenario + 1 # A, b, and t are observed, but not s
        self.statespace_dim = self.n_region + self.n_scenario + 1 + 1 # A, b, s, and t are all part of the state
        self.actionspace_dim = self.n_w_states


        # Precompute best portfolio choice for each (belief state, timestep).
        ## define quadrature approximation points and weights for the lognormal distribution of returns
        self.N_QUAD_1D = 20
        self.XI_1D, self.WI_1D = hermgauss(self.N_QUAD_1D)
        self.XI_3D = np.array(np.meshgrid(self.XI_1D, self.XI_1D, self.XI_1D)).T.reshape(-1, 3)  # (8000, 3)
        self.WI_3D = np.outer(np.outer(self.WI_1D, self.WI_1D).ravel(), self.WI_1D).ravel()      # (8000,)
        self.scaled_wi = self.WI_3D / np.sqrt(np.pi)**3  # scale weights for 3D quadrature

        self.best_weight_idx_matrix = self.build_best_weight_matrices()
        
        # initialize
        self.obs, self.state = self.reset()
        
    def reset(self, scenario=None):
        A = np.zeros(self.n_region)
        b = np.ones(self.n_scenario)/4
        if scenario is not None:
            s = [scenario]
        else:
            s = [np.random.choice(self.n_scenario)]
        self.s = s[0]
        t = [0]
        self.bidx = np.argmin(np.linalg.norm(self.b_states - b, axis=1))
        self.Aidx = np.argmin(np.linalg.norm(self.A_states - A, axis=1))
        self.state = np.concatenate([A, b, s, t])
        self.obs = np.concatenate([A, b, t])
        return self.obs, self.state
    
    def step(self, actionidx):
        '''
        award allocation step
        '''
        info = {}
        
        w = self.w_states[actionidx]
        b = self.state[self.sidx['b']]
        t = float(self.state[self.sidx['t']][0])

        # increment time step
        t_next = self.state[self.sidx['t']] + 1
        # define next portfolio
        A_next = t/(t+1) * self.state[self.sidx['A']] + 1/(t+1) * w
        dists = np.linalg.norm(self.A_states - A_next, axis=1)
        Aidx_next = np.argmin(dists)
        A_next = self.A_states[Aidx_next]

        # update belief about the scenario
        ## sample returns 
        mu = self.multi_timestep_returns[int(t_next), :, self.s]  # shape (3,)
        # lognormal returns
        log_returns = np.log(mu) - 0.5 * self.sig**2 + self.sig * np.random.randn(self.n_region)
        returns = np.exp(log_returns)
        likelihoods = np.ones(self.n_scenario)
        for s in range(self.n_scenario):
            mu_s = self.multi_timestep_returns[int(t_next), :, s]  # mean returns under scenario s
            for i in range(self.n_region):
                log_mu_i = np.log(mu_s[i]) - 0.5 * self.sig**2
                likelihoods[s] *= norm.pdf(np.log(returns[i]), loc=log_mu_i, scale=self.sig) / returns[i]
        b_next = b * likelihoods
        b_next = b_next / b_next.sum()
        # snap to nearest grid point
        dists = np.linalg.norm(self.b_states - b_next, axis=1)
        bidx_next = np.argmin(dists)
        b_next = self.b_states[bidx_next]

        # define reward and done

        if self.for_RL == 0:
            totreturns = (t+1) * np.dot(A_next, returns) if self.bnorm == 0 else np.dot(A_next, returns) # optionally normalize by budget to make rewards more comparable across settings with different budgets
            if self.gamma == 1:
                reward = np.log(totreturns)
            elif self.gamma < 1:
                reward =  (totreturns**(1 - self.gamma))/(1 - self.gamma) # use crra utliity function
            else: # gamma > 1, use log utility without the constant term.
                reward = (1-self.gamma) * np.log(totreturns) - np.log(self.gamma - 1)
        else: # for RL training, use the actual crra utility as the reward
            unscaled_totreturns = np.dot(A_next, returns)
            totreturns = (t+1)*unscaled_totreturns if self.bnorm == 0 else unscaled_totreturns 
            if self.gamma == 1:
                reward = np.log(totreturns)
            else:
                reward =  (totreturns**(1 - self.gamma))/(1 - self.gamma) # use crra utliity function
            reward = np.clip(reward, -self.R_clip, self.R_clip)
            reward = reward / self.absref_U
        info['portfolioreturn'] = totreturns
        
        done = t_next >= self.T
        # update state and observation
        self.bidx = bidx_next
        self.Aidx = Aidx_next
        self.state = np.concatenate([A_next, b_next, [self.s], t_next])
        self.obs = np.concatenate([A_next, b_next, t_next])
        return self.obs, reward, done, info

    def get_weights_index(self, weights):
        dists = np.linalg.norm(self.w_states - weights, axis=1)
        return np.argmin(dists)

    def best_weight(self, probidx, t):
        probs = self.b_states[probidx]
        returns_t = self.multi_timestep_returns[int(t)]
        #_, best_idx = self.max_utility_discrete(returns_t, probs, self.w_states)
        _, best_idx = self.max_utility_discrete(returns_t, probs, self.w_states)
        
        return best_idx


    def build_best_weight_matrices(self):
        '''
        Build a matrix of the best weight indices for each belief state and timestep.
        Each entry best_idx[probidx, t] gives the index of the weight vector in self.w_states
        that maximizes expected utility given belief state probidx at timestep t.
        To use, at timestep t you must use the current belief and t+1 to get the best weight for timestep t's allocation decision.
        '''
        bw = np.zeros((self.n_b_states, self.T + 1), dtype=int)
        for t in range(self.T + 1):
            mu = self.multi_timestep_returns[int(t)]                              # (R, S)
            log_mu = np.log(np.maximum(mu, 1e-10))
            log_r = (log_mu.T[:, None, :] - 0.5 * self.sig**2
                    + self.sig * np.sqrt(2) * self.XI_3D[None, :, :])             # (S, K, R)
            r = np.exp(log_r)
            port = np.einsum('ar,skr->ask', self.w_states, r)                     # (W, S, K)
            port = np.maximum(port, 1e-300)
            if np.isclose(self.gamma, 1.0):
                u = np.log(port)
            else:
                u = port ** (1 - self.gamma) / (1 - self.gamma)
            eu_per_s = u @ self.scaled_wi                                          # (W, S)
            eu_all   = eu_per_s @ self.b_states.T                                  # (W, B)
            bw[:, t] = np.argmax(eu_all, axis=0)
        return bw

    #def build_best_weight_matrices(self):
    #    '''
    #    Build a matrix of the best weight indices for each belief state and timestep.
    #    Each entry best_idx[probidx, t] gives the index of the weight vector in self.w_states
    #    that maximizes expected utility given belief state probidx at timestep t.
    #    To use, at timestep t you must use the current belief and t+1 to get the best weight for timestep t's allocation decision.
    #    '''
    #    best_idx = np.zeros((self.n_b_states, self.T + 1), dtype=int)
#
    #    for probidx in range(self.n_b_states):
    #        for t in range(self.T + 1):
    #            idx = self.best_weight(probidx, t)
    #            best_idx[probidx, t] = idx
    #    return best_idx

    def max_utility_discrete(self, returns, probs, w_states):
        port_returns = w_states @ returns  # (n_actions, n_scenarios)
        port_returns = np.maximum(port_returns, 1e-6)
        probs = np.asarray(probs, dtype=float)
        prob_sum = probs.sum()
        if prob_sum <= 0:
            probs = np.ones_like(probs) / probs.size
        else:
            probs = probs / prob_sum
        gamma = self.gamma
        
        if np.isclose(gamma, 1.0):
            log_utils = np.log(port_returns)
        else:
            log_utils = (1 - gamma) * np.log(port_returns)
        
        # log E[u] = log sum_s p_s * u_s, done in log space
        log_probs = np.full_like(probs, -np.inf, dtype=float)
        positive = probs > 0
        log_probs[positive] = np.log(probs[positive])
        log_probs = log_probs[np.newaxis, :]  # (1, n_scenarios)
        log_expected = logsumexp(log_utils[:,positive] + log_probs[:,positive], axis=1)  # (n_actions,)
        if np.isclose(gamma, 1.0) or gamma < 1.0:
            best = np.argmax(log_expected)
        else:
            best = np.argmin(log_expected)
        return log_expected[best], best

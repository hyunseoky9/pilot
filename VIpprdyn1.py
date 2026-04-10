import time
import pickle
from scipy.stats import norm
import numpy as np
from pprdyn1 import pprdyn1
from scipy.special import logsumexp
from numpy.polynomial.hermite import hermgauss

class VIpprdyn1(pprdyn1):
    def __init__(self, settings):
        super().__init__(settings)
        self.N_QUAD_1D = 20
        self.XI_1D, self.WI_1D = hermgauss(self.N_QUAD_1D)

        # Build 3D tensor product grid
        self.XI_3D = np.array(np.meshgrid(self.XI_1D, self.XI_1D, self.XI_1D)).T.reshape(-1, 3)  # (8000, 3)
        self.WI_3D = np.outer(np.outer(self.WI_1D, self.WI_1D).ravel(), self.WI_1D).ravel()      # (8000,)
        self.LOG_WI_3D = np.log(self.WI_3D)
        self.LOG_NORM_3D = -1.5 * np.log(np.pi)  # normalization: 1/pi^{3/2}

    
    def precompute_belief_transitions(self):
        '''
        Precompute the transition probabilities over the belief states so that
        the value iteration algorithm can efficiently update the value function
        over the belief space without recomputing the stochastic transitions
        at each iteration.
        '''
        print(f"Precomputing belief transitions (setting={self.settingID})...")
        # start timer
        
        start_time = time.time()

        btransmat = np.zeros((self.n_scenario, self.T, self.n_b_states, self.XI_3D.shape[0])) # transition matrix over belief states
        for s in range(self.n_scenario):
            for t in range(self.T):
                for i in range(self.n_b_states):
                    btransmat[s, t, i] = self.compute_belief_transition(s,i,t)
        
        # save numpy array of precomputed belief transitions
        np.save(f"belief_transitions_setting{self.settingID}.npy", btransmat)
        end_time = time.time()
        print(f"Precomputing belief transitions completed in {(end_time - start_time)/60:.2f} minutes.")

    def value_iteration(self):
        transmat = np.load(f"belief_transitions_setting{self.settingID}.npy").astype(int)
        
        policy = np.zeros((self.T, self.n_A_states, self.n_b_states), dtype=int)
        V = np.zeros((self.T + 1, self.n_A_states, self.n_b_states))
        
        # Precompute quadrature weights scaled by normalization
        scaled_wi = self.WI_3D * np.pi ** (-1.5)  # (8000,)
        
        for t in reversed(range(self.T)):
            print(f"t={t}")
            
            # Precompute returns per scenario: (n_scenario, 8000, 3)
            returns_per_s = np.zeros((self.n_scenario, self.XI_3D.shape[0], self.n_region))
            for s in range(self.n_scenario):
                mu_s = self.multi_timestep_returns[t + 1, :, s]
                returns_per_s[s] = self.compute_lognormal_returns(mu_s)
            
            # Precompute A_next_idx for all (a, w): (n_A, n_w)
            A_next_idx = np.zeros((self.n_A_states, self.n_w_states), dtype=int)
            for a in range(self.n_A_states):
                A_next_all = t / (t + 1) * self.A_states[a] + 1 / (t + 1) * self.w_states  # (n_w, 3)
                dists = np.linalg.norm(self.A_states[:, None, :] - A_next_all[None, :, :], axis=2)  # (n_A, n_w)
                A_next_idx[a] = np.argmin(dists, axis=0)  # (n_w,)
            
            # Precompute rewards for all (a, s): (n_A, n_scenario, 8000)
            port_returns = np.zeros((self.n_A_states, self.n_scenario, self.XI_3D.shape[0]))
            for s in range(self.n_scenario):
                port_returns[:, s, :] = (t + 1) * (self.A_states @ returns_per_s[s].T)  # (n_A, 8000)
            
            port_returns = np.maximum(port_returns, 1e-300)
            if self.gamma == 1:
                rewards_all = np.log(port_returns)
            else:
                rewards_all = port_returns ** (1 - self.gamma) / (1 - self.gamma)
            # rewards_all: (n_A, n_scenario, 8000)
            
            for a in range(self.n_A_states):
                a_next = A_next_idx[a]  # (n_w,)
                
                for b in range(self.n_b_states):
                    b_probs = self.b_states[b]  # (4,)
                    b_next_k = transmat[:, t, b, :]  # (n_scenario, 8000)
                    
                    # Gather rewards for all w: rewards_all[a_next, s, k]
                    # a_next: (n_w,), need (n_w, n_scenario, 8000)
                    r_w = rewards_all[a_next]  # (n_w, n_scenario, 8000)
                    
                    # Gather continuation values for all w: V[t+1, a_next[w], b_next_k[s, k]]
                    # a_next: (n_w,), b_next_k: (n_scenario, 8000)
                    # Result: (n_w, n_scenario, 8000)
                    cont_w = V[t + 1][a_next[:, None, None],
                                    b_next_k[None, :, :]]  # (n_w, n_scenario, 8000)
                    
                    # Integrand: reward + beta * continuation
                    integrand = r_w + self.beta * cont_w  # (n_w, n_scenario, 8000)
                    
                    # Weighted sum over quadrature points: (n_w, n_scenario)
                    expected_per_s = integrand @ scaled_wi  # (n_w, n_scenario)
                    
                    # Weight by belief over scenarios: (n_w,)
                    V_w = expected_per_s @ b_probs
                    
                    V[t, a, b] = np.max(V_w)
                    policy[t, a, b] = np.argmax(V_w)
            
            print(f"  t={t} done")
        
        # save 
        output = {'policy': policy, 'V': V}
        with open(f'VI_policy_setting{self.settingID}.pkl', 'wb') as f:
            pickle.dump(output, f)
        return policy, V[:self.T]

    def compute_belief_transition(self, s, i, t):
        '''
        Compute the probability of transitioning from belief state i to belief state j
        at time t given the dynamics of the environment. This function should return
        a probability vectors.
        '''
        mu_s = self.multi_timestep_returns[t+1,:,s]
        r_points = self.compute_lognormal_returns(mu_s) # (8000, 3)
        # update belief state j given returns r_points
        
        likelihoods = np.ones((self.n_scenario , r_points.shape[0])) # (n_scenario, n_quadrature_points)
        for i_s in range(self.n_scenario):
            mu_hyp = self.multi_timestep_returns[t+1,:,i_s]
            log_mu_hyp = np.log(mu_hyp) - 0.5 * self.sig**2
            likelihoods[i_s] *= np.prod(norm.pdf(np.log(r_points), loc=log_mu_hyp, scale=self.sig) / r_points, axis=1)
        b = self.b_states[i]
        b_next = b[:,np.newaxis] * likelihoods # (n_scenario, n_quadrature_points)
        b_next = b_next / np.maximum(b_next.sum(axis=0),1e-300) # snap to nearest grid point
        dists = np.linalg.norm(self.b_states[:, np.newaxis, :] - b_next.T[np.newaxis, :, :], axis=2)
        bidx_next = np.argmin(dists,axis=0)
        return bidx_next

    def compute_lognormal_returns(self,mu_s):
        """
        Compute lognormal return samples at all quadrature points.
        mu_s: (3,) mean returns per region under scenario s
        Returns: (8000, 3) array of return values at each quadrature point
        
        Each region i: r_i = exp(log(mu_i) - 0.5*sigma^2 + sigma*sqrt(2)*xi)
        so that E[r_i] = mu_i
        """
        log_mu = np.log(np.maximum(mu_s, 1e-10))
        log_r = log_mu[np.newaxis, :] - 0.5 * self.sig**2 + self.sig * np.sqrt(2) * self.XI_3D
        return np.exp(log_r)

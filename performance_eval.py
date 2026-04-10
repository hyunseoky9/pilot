# write a performance evaluation function to track cumulative reward over episodes
from scipy.special import logsumexp
from pprdyn1 import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
import numpy as np
import pandas as pd
from utilitysolver import utilitysolver

#from VI_pprdyn1 import _act, build_optimal_controller_fully_observable
#import torch
def avgperformance(env, config, policy_printout=False,printout = False):
    #device = torch.device('cpu')  # Force CPU usage
    policytype = config['policytype']
    num_episodes = config['num_episodes']
    if policytype == 0: # Value iteration
        print( 'policy type: value iteration')
        with open(f'./value_iteration/VI_controller_setting{config["VIenvsetting"]}.pkl', 'rb') as f:
            ctrl = pickle.load(f)
        Policy = ctrl['policy']
        # reinitiate the environment
        settings = ctrl['envinfo']
        print(settings)
        env = pprdyn1(settings)
    elif policytype == 1: # rolling E utility optimization on next year's returns
        print( 'policy type: rolling E utility optimization on next year returns')
    elif policytype == 2: # static E utility optimization on final timestep not updating the belief(all mpt papers)
        print( 'policy type: static E utility optimization on final timestep not updating the belief')
    elif policytype == 3: # rolling E utility optimization on final timestep updating the belief probability on the scenarios
        print( 'policy type: rolling E utility optimization on final timestep updating the belief probability on the scenarios')
    elif policytype == 4: # rolling E utility optimization on next year's returns not updating the belief
        print( 'policy type: static E utility optimization on next year returns not updating the belief')
    elif policytype == 5: # VI policy trained with constant belief.
        

    if policytype in [1,2,3,4]:
        best_wmat = env.build_best_weight_matrices() # best allocation matrices for each belief state and timestep

    rewards = []
    #allocations = np.zeros((num_episodes, env.T, env.n_region))
    #portfolios = np.zeros((num_episodes, env.T, env.n_region))
    betas = env.beta ** np.arange(env.T)
    log_betas = np.log(env.beta) * np.arange(env.T)
    K = np.sum(betas)
    logK = np.log(K)
    for i in range(num_episodes):
        obs, state = env.reset()
        if printout:
            print(f'episode: {i+1}')
            print(f'scenario: {env.s}')
        input = obs.copy()
        done = False
        ep_reward = []
        tt=0
        while not done:
            if policytype == 0:
                actionidx = _act(env, Policy, input)
            elif policytype == 1:
                actionidx = best_wmat[env.bidx, int(env.state[env.sidx['t']])+1]
            elif policytype == 2:
                actionidx = best_wmat[env.b_states.shape[0]-1, -1]
            elif policytype == 3:
                actionidx = best_wmat[env.bidx, -1]
            elif policytype == 4:
                actionidx = best_wmat[env.b_states.shape[0]-1, int(env.state[env.sidx['t']])+1]
            #allocations[i,int(env.state[env.sidx['t']])] = env.w_states[actionidx]
            #portfolios[i,int(env.state[env.sidx['t']])] = env.state[env.sidx['A']]
            if printout:
                print(f't: {env.state[env.sidx["t"]]}, allocation: {env.w_states[actionidx]}')
            tt += 1
            obs, reward, done, info = env.step(actionidx)
            input = env.obs.copy()
            ep_reward.append(reward)
        ep_reward = np.array(ep_reward)
        if env.gamma <= 1:
            totep_reward = np.sum(ep_reward*betas)
        else: 
            totep_reward = logsumexp(ep_reward + log_betas) # log total reward over episodes
        rewards.append(totep_reward)
        if (i+1) % 1000 == 0:
            print(f'Episode {i+1} done')
    # calculate certainty equivalent across episode rewards
    if env.gamma <1:
        V = np.mean(rewards)
        ce = ((1- env.gamma)*V/K)**(1/(1 - env.gamma))
    elif env.gamma == 1:
        V = np.mean(rewards)
        ce = np.exp(V/K)
    else: 
        m = np.max(rewards)
        log_negV = np.log(1/len(rewards) * np.sum(np.exp(rewards - m))) + m
        print(f"MC log_negV = {log_negV:.6f}")
        logA = np.log(env.gamma - 1) + log_negV - logK
        logce = logA / (1- env.gamma)
        ce = np.exp(logce)
    summary = {'rewards': rewards}  # 'allocations': allocations, 'portfolios': portfolios}
    
    
    print(f'Average reward over {num_episodes} episodes: {np.mean(rewards):.4f}, certrainty equivalent: {ce:.4f}')
    return summary

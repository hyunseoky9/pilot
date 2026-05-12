from call_paramset import call_env
import numpy as np
import torch
from scipy.special import logsumexp
import copy

def calc_performance(env, device, rms, fstack, policy, episodenum=1000, t_maxstep = 1000, deterministic_eval=False):
    """
    same as calc_performance.py but built for PPO algorithms. 
    non-parallelized version.
    calculate the performance of the agent in the environment.
    """
    print('serial calc_performance called')
    avgrewards = 0
    env_eval = copy.deepcopy(env)
    env_eval.for_RL = 0 # use general evaluation reward function for evaluation, even if the environment was set up with for_RL=1 for training.
    rewards = []
    betas = env.beta ** np.arange(env.T)
    log_betas = np.log(env.beta) * np.arange(env.T)
    K = np.sum(betas)
    logK = np.log(K)

    for i in range(episodenum):
        ep_reward = []
        env_eval.reset()
        stack = np.concatenate([rms.normalize(env_eval.obs)]*fstack) if rms is not None else np.concatenate([env_eval.obs]*fstack)
        done = False
        t = 0
        while done == False:
            state = rms.normalize(env_eval.obs) if rms is not None else env_eval.obs
            stack = np.concatenate((stack[len(state):], state))

            with torch.no_grad():                                   # <– no grads here
                s = torch.as_tensor(stack.copy(), dtype=torch.float32, device=device).unsqueeze(0)
                if deterministic_eval:
                    action = policy.get_deterministic_action(s)
                else:
                    action = policy.getaction(s, get_action_only=True)
                action = action.cpu().numpy().squeeze(0)
            _, reward, done, _ = env_eval.step(action)
            ep_reward.append(reward)
            if t >= (t_maxstep - 1):
                done = True
            t += 1
        if env.gamma <= 1:
            totep_reward = np.sum(ep_reward*betas)
        else: 
            totep_reward = logsumexp(ep_reward + log_betas) # log total reward over episodes
        rewards.append(totep_reward)
    # calculate certainty equivalent across episode rewards
    if env.gamma <1:
        V = np.mean(rewards)
        ce = ((1- env.gamma)*V/K)**(1/(1 - env.gamma))
    elif env.gamma == 1:
        V = np.mean(rewards)
        ce = np.exp(V/K)
    else:
        # log-sum-exp trick, specifically the max-shift stabilization trick.
        m = np.max(rewards)
        log_negV = np.log(1/len(rewards) * np.sum(np.exp(rewards - m))) + m
        logA = np.log(env.gamma - 1) + log_negV - logK
        logce = logA / (1- env.gamma)
        ce = np.exp(logce)
    return ce
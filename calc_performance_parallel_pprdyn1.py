from call_paramset import call_env
import random
import numpy as np
import torch
import torch.multiprocessing as mp
def calc_performance_parallel(env, device, seed, config, rms, fstack, policy=None, episodenum=1000,
                                t_maxstep=1000, deterministic_eval=False):
    """
    same as calc_performance-parallel.py but built for PPO algorithms. 
    parallelized version
    calculate the performance of the agent in the environment.
    """
    print('parallel calc_performance called')
    action_size = len(env.actionspace_dim)
    # set environment configuration for the workers
    initlist = None
    #config = "{'init': %s,'paramset': %d, 'discretization': %d}" %(str(initlist),env.parset+1,env.discset)
    envinit_params = {'envconfig': config, 'envid': env.envID}

    # worker parameters
    #################
    num_workers = 4
    #################
    if policy is not None:
        policy.share_memory()
    total_rewards = mp.Value('f', 0.0)


    # Set multiprocessing method
    mp.set_start_method("spawn", force=True)
    # start timer
    processes = []
    # divide the number of epsiodes by workers.
    episodenum_perworker = [int(np.floor(episodenum/num_workers))]*(num_workers - 1)
    episodenum_perworker = episodenum_perworker + [int(episodenum - np.sum(episodenum_perworker))]
    # Spawn worker processes
    for worker_id in range(num_workers):
        p = mp.Process(target=worker, args=(policy, episodenum_perworker[worker_id],
                                            rms, worker_id, envinit_params,
                                            t_maxstep, fstack, device,
                                            total_rewards,seed, deterministic_eval))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()

    return total_rewards.value/episodenum




def worker(policy, workerepisodenum, rms, worker_id, envinit_params, t_maxstep, fstack,
            device, total_rewards,seed, deterministic_eval):
    # set seed
    worker_seed = seed + (worker_id + 1)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    # set env
    env = call_env(envinit_params)

    totrewards_perworker = 0 # set reward counter.

    for i in range(workerepisodenum):
        rewards = 0
        env.reset()
        stack = np.concatenate([rms.normalize(env.obs)]*fstack) if rms is not None else np.concatenate([env.obs]*fstack)
        done = False
        t = 0
        while done == False:
            state = rms.normalize(env.obs) if rms is not None else env.obs
            stack = np.concatenate((stack[len(state):], state))

            with torch.no_grad():                                   # <– no grads here
                s = torch.as_tensor(stack.copy(), dtype=torch.float32, device=device).unsqueeze(0)
                if deterministic_eval:
                    action = policy.get_deterministic_action(s)
                else:
                    action = policy.get_action(s, get_action_only=True)
                action = action.cpu().numpy().squeeze(0)
            _, reward, done, _ = env.step(action)
            rewards += reward
            if t >= (t_maxstep - 1):
                done = True
            t += 1
        totrewards_perworker += rewards
        
    # upload total rewards    
    with total_rewards.get_lock(): 
        total_rewards.value += totrewards_perworker # update global counter

    
# PPO2 training script
import sys
from PPO import *
from call_paramset import call_paramset, call_env
import pandas as pd
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np

if __name__ == "__main__":
    paramid = int(sys.argv[1])
    iteration_num = 20
    hyperparameterization_set_filename = './hyperparamsets/PPO2hyperparamsets.csv'
    paramdflist = call_paramset(hyperparameterization_set_filename,paramid)
    tuneset = 1
    best_scores = []
    seeds = []
    for paramdf in paramdflist: 
        for iteration in range(iteration_num):
            print('-------------------------------------------------------\n--------------------------------------------------------')
            seed = paramdf['seed'] # make sure iteration_num matches with the number of seeds if seeds are specified
            if seed == 'random':
                seednum = random.randint(0,1000000)
            else:
                seednum = int(seed)
            os.environ["PYTHONHASHSEED"] = str(seednum)
            random.seed(seednum)
            np.random.seed(seednum)
            torch.manual_seed(seednum)
            # environment configuration
            env = call_env(paramdf)
            meta = {'paramid': paramid, 'iteration': iteration, 'seed': seednum}
            agent = PPO(env, paramdf,meta)
            _, scores, training_info = agent.train()
            best_scores.append(np.max(scores))
            seeds.append(seednum)
                
            print(f"Tuneset {tuneset}, Iteration {iteration}: {scores}")
        tuneset += 1
    seedsNscores = pd.DataFrame({'seed': seeds, 'best_score': best_scores})
    print(seedsNscores.sort_values(by='best_score', ascending=False))
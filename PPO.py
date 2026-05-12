import pickle
import numpy
from ppoagent import PPOAgent
from setup_logger import setup_logger
import shutil
import torch
import os
import numpy as np
from calc_performance_pprdyn1 import calc_performance
from calc_performance_parallel_pprdyn1 import calc_performance_parallel
from ppo_actor import Actor_pprdyn1
from ppo_critic import Critic
import random
from FixedMeanStd import FixedMeanStd
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, MultiStepLR, CosineAnnealingLR


class PPO():
    def __init__(self, env, paramdf, meta):
        """
        PPO agent.
        env: Environment object
        paramdf: DataFrame containing hyperparameters for the agent
        meta: metadata (paramid, iteration, seed) for logging and saving results
        """
        self.algorithmID = 'PPO'
        self.paramdf = paramdf
        # define parameters
        ## env setup
        self.env = env
        ## meta info
        self.paramid = meta['paramid']
        self.iteration = meta['iteration']
        self.seed = meta['seed']
        ## define the path for the new directory
        self.parent_directory = "./PPO_results/"
        self.new_directory = f'seed{self.seed}_paramid{self.paramid}'
        self.path = os.path.join(self.parent_directory, self.new_directory)
        ## set path 
        os.makedirs(self.path, exist_ok= True)
        self.testwd = f'./PPO_results/{self.new_directory}'
        self.logger = setup_logger(self.testwd) ## set up logging

        # Device selection
        # options from paramdf['device']: {'cuda','gpu','cpu','mps'}
        requested_device = (
            str(self.paramdf['device']).lower()
            if isinstance(self.paramdf, dict) and 'device' in self.paramdf
            else 'auto'
        )

        if requested_device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        elif requested_device in ('cuda', 'gpu'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif requested_device == 'mps':
            self.device = torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        print(f"Using {self.device.type} device")

        # parameters
        ## NN parameters
        self.state_size = self.env.obsspace_dim 
        self.action_size = self.env.actionspace_dim
        self.actor_hidden_num = int(paramdf['actor_hidden_num']) # number of hidden layers in the actor network
        self.actor_hidden_size = eval(paramdf['actor_hidden_size']) # size of hidden layers in the actor network
        self.critic_hidden_num = int(paramdf['critic_hidden_num']) # number of hidden layers in the critic network for trunk
        self.critic_hidden_size = eval(paramdf['critic_hidden_size']) # size of hidden layers in the critic network for trunk

        ## training parameters
        self.advantage_normalization = bool(int(paramdf['advantage_normalization'])) # whether to normalize advantages
        self.rolloutlen = int(paramdf['rollout_len']) # e.g., 20
        self.minibatch_size = int(paramdf['minibatch_size']) # e.g., 5 
        self.n_epochs = int(paramdf['n_epochs']) # e.g., 4
        self.max_steps = int(paramdf['max_steps']) # maximum steps per episode, e.g., 1000
        self.episodenum = int(paramdf['episodenum'])

        ## learning rates
        self.actor_lr = float(paramdf['actor_lr']) # learning rate for actor network
        self.critic_lr = float(paramdf['critic_lr']) # learning rate for critic network
        self.actor_lrdecayrate = float(paramdf['actor_lrdecay']) # learning rate decay rate for actor network
        self.critic_lrdecayrate = float(paramdf['critic_lrdecay']) # learning rate decay rate for critic network
        if paramdf['actor_minlr'] == 'inf':
            self.actor_min_lr = float('-inf') # minimum learning rate for actor network
        else:
            self.actor_min_lr = float(paramdf['actor_minlr'])
        if paramdf['critic_minlr'] == 'inf':
            self.critic_min_lr = float('-inf') # minimum learning rate for critic network
        else:
            self.critic_min_lr = float(paramdf['critic_minlr'])
        self.actor_lrdecaytype = paramdf['actor_lrdecaytype'] # learning rate decay type for actor network
        self.critic_lrdecaytype = paramdf['critic_lrdecaytype'] # learning rate decay type for critic network
        self.scheduler_info = eval(paramdf['scheduler_info'])

        ## standardize
        self.standardize = bool(int(paramdf['standardize'])) # whether to standardize the advantages

        ## loss coefficients 
        self.c1 = float(paramdf['c1']) # coefficient for value function loss
        self.c2 = float(paramdf['c2']) # coefficient for entropy bonus
        self.entropy_loss_included = bool(int(paramdf['entropy_loss_included'])) # whether to include entropy loss in the total loss
        self.policy_clip = float(paramdf['policy_clip']) # clipping parameter for PPO
        self.KL_stopping= bool(int(paramdf['KL_stopping'])) # whether to use KL divergence stopping criterion
        self.target_KL = float(paramdf['target_KL']) # target KL divergence for adaptive KL penalty (if used)

        ## discounting and GAE lambda
        self.gamma = float(paramdf['gamma']) # discount factor
        self.gae_lambda = float(paramdf['gae_lambda']) # GAE lambda parameter

        ## evaluation parameters
        self.evaluation_interval = int(paramdf['evaluation_interval']) # evaluate every n episodes
        self.performance_sampleN = int(paramdf['performance_sampleN']) # number of episodes to sample for performance evaluation
        self.parallel_testing = bool(int(paramdf['parallel_testing'])) # whether to use parallel testing for performance evaluation
        self.deterministic_eval = bool(int(paramdf['deterministic_eval'])) # whether to use deterministic policy for evaluation

        # print out the parameters
        print(f'paramID: {self.paramid}, iteration: {self.iteration}, seed: {self.seed}')
        print(f"env: {self.env.envID}")
        print(f"device: {self.device}")
        print(f"state size: {self.state_size}, action size: {self.action_size}")
        print(f"actor: hidden num: {self.actor_hidden_num}, hidden size: {self.actor_hidden_size}")
        print(f"critic: hidden num: {self.critic_hidden_num}, hidden size: {self.critic_hidden_size}")
        print(f"actor lr: {self.actor_lr}, actor lr decay rate: {self.actor_lrdecayrate}, actor min lr: {self.actor_min_lr}")
        print(f"critic lr: {self.critic_lr}, critic lr decay rate: {self.critic_lrdecayrate}, critic min lr: {self.critic_min_lr}")
        print(f"standardize: {self.standardize}")
        print(f"gamma: {self.gamma}, gae_lambda: {self.gae_lambda}")
        print(f'advantage normalization: {self.advantage_normalization}')
        print(f'KL stopping: {self.KL_stopping}, target KL: {self.target_KL}')
        print(f"c1: {self.c1}, c2: {self.c2}, entropy_loss_included: {self.entropy_loss_included}, policy_clip: {self.policy_clip}")
        print(f'rollout length: {self.rolloutlen}, minibatch size: {self.minibatch_size}, n_epochs: {self.n_epochs}')
        print(f"evaluation interval: {self.evaluation_interval}, performance sampleN: {self.performance_sampleN}, parallel testing: {self.parallel_testing}, deterministic eval: {self.deterministic_eval}")
        print(f"max steps: {self.max_steps}, episodenum: {self.episodenum}")

        # create agent
        if 'metapop1' in self.env.envID:
            if self.env.partial_observability == 0:
                extrainfo = {'npatches': self.env.patchnum, 'kR': self.env.kR, 'kS': self.env.kS}
                actionsize = self.action_size
                if self.env.kR < self.env.patchnum:
                    extrainfo['Rheadsize'] = self.env.patchnum + 1 # restoration head size.
                    extrainfo['Rbernoulli'] = 0
                    actionsize += 1 # add 1 for STOP token when sampling
                else:
                    extrainfo['Rheadsize'] = self.env.patchnum
                    extrainfo['Rbernoulli'] = 1
                if self.env.kS < self.env.patchnum:
                    extrainfo['Sheadsize'] = self.env.patchnum + 1 # supplementation head size
                    extrainfo['Sbernoulli'] = 0
                    actionsize += 1 # add 1 for STOP token when sampling
                else: 
                    extrainfo['Sheadsize'] = self.env.patchnum
                    extrainfo['Sbernoulli'] = 1
                actor = Actor_metapop1_MDP(self.state_size, actionsize, # add 1 for the beta parameter, which is the first output, and the rest are for the dirichlet distribution (total of 5 outputs for 4 actions)
                                            self.actor_hidden_size, self.actor_hidden_num,
                                            self.actor_lrdecayrate, self.actor_lr,
                                            self.actor_min_lr, self.actor_lrdecaytype, 
                                            self.scheduler_info, self.device, self.entropy_loss_included, extrainfo)
                critic = Critic(self.state_size, 
                                self.critic_hidden_size, self.critic_hidden_num,
                                self.critic_lrdecayrate, self.critic_lr, 
                                self.critic_min_lr, self.critic_lrdecaytype, 
                                self.scheduler_info, self.device)
                # create agent
                self.agent = PPOAgent(c1=self.c1, c2=self.c2, entropy_loss=self.entropy_loss_included,  # loss coefficients
                                minibatch_size=self.minibatch_size,  # minibactch size
                                policy_clip=self.policy_clip, # PPO clipping parameter
                                gamma=self.gamma, gae_lambda=self.gae_lambda, # discount factor and GAE lambda
                                n_epochs=self.n_epochs, # number of epochs for updating the policy
                                adv_normalization=self.advantage_normalization, # whether to normalize advantages
                                KL_stopping=self.KL_stopping, target_KL=self.target_KL, # whether to use KL stopping and the target KL value
                                actor=actor, critic=critic) # actor and critic networks
                
        elif 'pprdyn1' in self.env.envID: # FOR NOW just use the same actor architecture for all envs, but can change this in the future if needed
            actor = Actor_pprdyn1(self.state_size, self.action_size,
                                        self.actor_hidden_size, self.actor_hidden_num,
                                        self.actor_lrdecayrate, self.actor_lr,
                                        self.actor_min_lr, self.actor_lrdecaytype, 
                                        self.scheduler_info, self.device, self.entropy_loss_included,None)

            critic = Critic(self.state_size, 
                            self.critic_hidden_size, self.critic_hidden_num,
                            self.critic_lrdecayrate, self.critic_lr, 
                            self.critic_min_lr, self.critic_lrdecaytype, 
                            self.scheduler_info, self.device)

            # create agent
            self.agent = PPOAgent(c1=self.c1, c2=self.c2, entropy_loss=self.entropy_loss_included,  # loss coefficients
                            minibatch_size=self.minibatch_size,  # minibactch size
                            policy_clip=self.policy_clip, # PPO clipping parameter
                            gamma=self.gamma, gae_lambda=self.gae_lambda, # discount factor and GAE lambda
                            n_epochs=self.n_epochs, # number of epochs for updating the policy
                            adv_normalization=self.advantage_normalization, # whether to normalize advantages
                            KL_stopping=self.KL_stopping, target_KL=self.target_KL, # whether to use KL stopping and the target KL value
                            actor=actor, critic=critic) # actor and critic networks

        # create standarization tool
        self.rms = FixedMeanStd(env) if self.standardize else None
        #self.rms = RunningMeanStd(len(env.obsspace_dim), self.max_steps) if self.standardize else None


    def train(self):
        best_score = 0
        score_history = []
        self.did_first_update = False # flag to indicate if the first update has been done, used for learning rate scheduler stepping
        learn_iters = 0
        avg_score = 0
        n_steps = 0
        inttestscores = [] # interval test scores
        for i_episode in range(1, self.episodenum + 1):
            observation = self.env.reset()
            done = False
            score = 0
            episteps = 0
            while not done:
                # STANDARDIZE OBS
                observation = self.rms.normalize(self.env.obs) if self.standardize else self.env.obs
                with torch.no_grad():
                    action, prob, val, ainfo = self.agent.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                if self.standardize: # standardize
                    self.rms.stored_batch.append(observation_) # store the state for running mean std calculation
                    if self.rms.rolloutnum >= self.rms.updateN:
                        self.rms.update()
                    self.rms.rolloutnum += 1
                n_steps += 1
                score += reward
                self.agent.remember(observation, action, prob, val, reward, done, ainfo)
                if n_steps % self.rolloutlen == 0:
                    self.agent.learn()
                    if not self.did_first_update:
                        self.did_first_update = True
                    # step the learning rate schedulers if using exponential decay
                    if isinstance(self.agent.actor.scheduler, ExponentialLR):
                        self.agent.actor.scheduler.step() # Decay the learning rate
                    if isinstance(self.agent.critic.scheduler, ExponentialLR):
                        self.agent.critic.scheduler.step() # Decay the learning rate
                    learn_iters += 1
                episteps += 1
                if episteps >= self.max_steps:
                    break
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            # step the learning rate schedulers if using multistep or cosine annealing
            if self.did_first_update:
                if isinstance(self.agent.actor.scheduler, (MultiStepLR, CosineAnnealingLR)):
                    self.agent.actor.scheduler.step()
                if isinstance(self.agent.critic.scheduler, (MultiStepLR, CosineAnnealingLR)):
                    self.agent.critic.scheduler.step()

            # print out episode information every interval
            if i_episode % 500 == 0:
                print(f"Episode {i_episode}")

            # evaluate at every specified interval episodes
            if i_episode % self.evaluation_interval == 0: 
                if self.parallel_testing:
                    inttestscore = calc_performance_parallel(self.env, self.device, self.seed, self.paramdf['envconfig'], self.rms, 1, self.agent.actor, self.performance_sampleN, self.max_steps, self.deterministic_eval)
                else:
                    inttestscore = calc_performance(self.env,self.device,self.rms,1,self.agent.actor,self.performance_sampleN,self.max_steps,self.deterministic_eval)
                inttestscores.append(inttestscore)
                actorpath = f"{self.testwd}/PolicyNetwork_{self.env.envID}_par{self.env.paramsetID}_set{self.env.settingID}_{self.algorithmID}_episode{i_episode}.pt"
                criticpath = f"{self.testwd}/ValueNetwork_{self.env.envID}_par{self.env.paramsetID}_set{self.env.settingID}_{self.algorithmID}_episode{i_episode}.pt"
                self.agent.save_models(actorpath, criticpath)
            
                # save the running mean and sd/var as well for this episode in pickle
                if self.standardize:
                    with open(f"{self.testwd}/rms_{self.env.envID}_par{self.env.paramsetID}_set{self.env.settingID}_{self.algorithmID}_episode{i_episode}.pkl", "wb") as file:
                        pickle.dump(self.rms, file)
                
                critic_current_lr = self.agent.critic.optimizer.param_groups[0]['lr']
                actor_current_lr = self.agent.actor.optimizer.param_groups[0]['lr']
                print(f"Episode {i_episode}, Learning Rate: A{np.round(actor_current_lr, 6)}/C{np.round(critic_current_lr, 6)} Avg Performance: {inttestscore:.2f}")
                print('-----------------------------------')

            #if avg_score > best_score:
            #    best_score = avg_score
            #    print(f"(New best avg (last 100 epi's) score: {best_score:.1f} at episode {i_episode})")


        ## save best model
        bestidx = np.array(inttestscores).argmax()
        bestfilename = f"{self.testwd}/PolicyNetwork_{self.env.envID}_par{self.env.paramsetID}_set{self.env.settingID}_{self.algorithmID}_episode{(bestidx+1)*self.evaluation_interval}.pt"
        print(f'best Policy network found at episode {(bestidx+1)*self.evaluation_interval}')
        shutil.copy(bestfilename, f"{self.testwd}/bestPolicyNetwork_{self.env.envID}_par{self.env.paramsetID}_set{self.env.settingID}_{self.algorithmID}.pt")
        ## save rms for best model
        if self.standardize:
            best_rms_filename = f"{self.testwd}/rms_{self.env.envID}_par{self.env.paramsetID}_set{self.env.settingID}_{self.algorithmID}_episode{(bestidx+1)*self.evaluation_interval}.pkl"
            shutil.copy(best_rms_filename, f"{self.testwd}/bestRMS_{self.env.envID}_par{self.env.paramsetID}_set{self.env.settingID}_{self.algorithmID}.pkl")

        ## save performance
        np.save(f"{self.testwd}/rewards_{self.env.envID}_par{self.env.paramsetID}_set{self.env.settingID}_{self.algorithmID}.npy", inttestscores)

        ## lastly save the configuration.
        param_file_path = os.path.join(self.testwd, f"config.txt")
        with open(param_file_path, 'w') as param_file:
            for key, value in self.paramdf.items():
                param_file.write(f"{key}: {value}\n")
        sorted_scores = np.sort(inttestscores)[::-1]

        print("Training completed.")
        
        return self.agent.actor, sorted_scores
    
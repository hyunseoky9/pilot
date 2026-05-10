import os 
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim


class PPOMemory:
    def __init__(self, minibatch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.info = []

        self.minibatch_size = minibatch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.minibatch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.minibatch_size] for i in batch_start]

        return np.array(self.states), \
                np.array(self.actions), \
                np.array(self.probs), \
                np.array(self.vals), \
                np.array(self.rewards), \
                np.array(self.dones), \
                self.info, \
                batches
    
    def store_memory(self, state, prob, val, action, reward, done, info=None):
        self.states.append(state)
        self.probs.append(prob)
        self.vals.append(val)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.info.append(info)
    
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.info = []

class PPOAgent:
    def __init__(self, c1, c2, entropy_loss, 
                 minibatch_size,
                 policy_clip,
                 gamma, gae_lambda,
                 n_epochs,
                 adv_normalization,
                 KL_stopping, target_KL,
                 actor, critic):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.c1 = c1
        self.c2 = c2
        self.entropy_loss = entropy_loss
        self.actor = actor
        self.critic = critic
        self.KL_stopping = KL_stopping
        self.target_KL = target_KL
        self.memory = PPOMemory(minibatch_size)
        self.adv_normalization = adv_normalization

    def remember(self, state, action, probs, vals, reward, done, info=None):
        self.memory.store_memory(state, probs, vals, action, reward, done, info)
    
    def save_checkpoint(self, network,path):
        T.save(network, path)

    def save_models(self,actorpath,criticpath):
        self.save_checkpoint(self.actor, actorpath)
        #self.save_checkpoint(self.critic, criticpath)

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).unsqueeze(0).to(self.actor.device)
        
        action, logprob, info = self.actor.getaction(state)
        value = self.critic(state)
    
        probs = T.squeeze(logprob).item()
        action = T.squeeze(action).cpu().detach().numpy()
        value = T.squeeze(value).item()

        return action, probs, value, info

    def compute_gae_1d(self, rewards,values,dones,gamma ,lam ,last_value=0.0):
        """
        O(T) GAE(λ) for a single trajectory or a concatenated rollout.

        Args:
            rewards: shape [T]
            values:  shape [T]  (V(s_t) stored during rollout)
            dones:   shape [T]  (1 if terminal at t else 0)
            gamma: discount factor
            lam: GAE lambda
            last_value: V(s_{T}) for bootstrapping the final step if not terminal.
                        If you don't have V(s_T), pass 0 and this still behaves sensibly.

        Returns:
            advantages: shape [T]
            returns:    shape [T] where returns = advantages + values (GAE-style target)
        """
        T_len = len(rewards)
        advantages = np.zeros(T_len, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(T_len)):
            nonterminal = 1.0 - float(dones[t])

            next_value = last_value if t == T_len - 1 else values[t + 1]

            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            gae = delta + gamma * lam * nonterminal * gae
            advantages[t] = gae

            # If this timestep ended an episode, reset the accumulator
            if dones[t]:
                gae = 0.0

        returns = advantages + values.astype(np.float32)
        return advantages, returns

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, done_arr, info_arr, batches = \
                self.memory.generate_batches()
            
            values = vals_arr
            last_value = 0
            advantages, returns = self.compute_gae_1d(reward_arr,values,done_arr,
                                                      self.gamma,self.gae_lambda,last_value)
            #advantage = np.zeros(len(reward_arr), dtype=np.float32)
            #for t in range(len(reward_arr)- 1):
            #    discount = 1
            #    a_t = 0
            #    for k in range(t, len(reward_arr) - 1):
            #        a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1 - int(done_arr[k])) - values[k])
            #        discount *= self.gamma * self.gae_lambda
            #    advantage[t] = a_t
            #advantage = T.tensor(advantage).to(self.actor.device)
            # Advantage normalization (once per epoch, before minibatches)
            if self.adv_normalization:
                advantages = (advantages - advantages.mean()) / (advantages.std(ddof=0) + 1e-10)

            kl_running = 0.0
            kl_count = 0
            stop_early = False

            for batch in batches:
                # convert batch data to tensors
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                # calculate critic value with new network
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)
                # calculate new log probs with new network
                new_probs, current_entropy = self.actor.get_log_prob(states, actions, (info_arr, batch))
                prob_ratio = (new_probs - old_probs).exp()


                # KL estimate
                approx_kl = (old_probs - new_probs).mean()
                kl_running += approx_kl.item()
                kl_count += 1
                # get advantages and returns for the minibatch and convert to tensors
                advantage = T.tensor(advantages[batch], dtype=T.float, device=self.actor.device)
                returns_t = T.tensor(returns[batch], dtype=T.float, device=self.actor.device)
                # calculate actor loss
                weighted_probs = advantage * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                # calculate critic loss
                critic_loss = (returns_t - critic_value).pow(2).mean()
                # calculate total loss
                total_loss = actor_loss + self.c1 * critic_loss
                # add entropy loss if using
                if self.entropy_loss:
                    entropy_loss = -self.c2 * current_entropy.mean()
                    total_loss += entropy_loss
                # take a gradient step
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.actor.optimizer.step()
                self.critic.optimizer.step()


                # KL early stopping check 
                if (self.KL_stopping) and (approx_kl.item() > self.target_KL):
                    stop_early = True
                    break
            if stop_early:
                break


        self.memory.clear_memory() # clear memory after learning is done before next round of data collection

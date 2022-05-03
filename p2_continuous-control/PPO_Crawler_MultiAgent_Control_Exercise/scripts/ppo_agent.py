import numpy as np
import random
import copy
from collections import namedtuple, deque
from model import PPO_ActorCritic
import torch
import torch.nn.functional as F
import torch.optim as optim
from params import Params
from replay_buffer import ReplayBuffer
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class PPO_Agent():
    """
    PyTorch Implementation of Proximal Policy Optimization (PPO) with A3C/GAE Value Estimation:
    """
    
    def __init__(self, state_size, action_size, params=Params()):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            total_state_size (int): dimension of each state
            action_size (int): dimension of each action
        """

        # torch.autograd.set_detect_anomaly(True)         

        self.params = params
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(self.params.random_seed)
        self.device = self.params.device

        # Actor Network & Critic Network (Views entire state space)
        # NOTE: Optimizes actor-critic net through joint loss function
        self.ppo_ac_net = PPO_ActorCritic(state_size, action_size, self.params).to(self.device)
        self.optimizer = optim.Adam(self.ppo_ac_net.parameters(), lr=self.params.lr, 
                                    eps=self.params.optimizer_eps, weight_decay=self.params.weight_decay)

        # Others
        self.memory = ReplayBuffer(state_size, action_size, self.params.random_seed, params)
        self.eps = self.params.eps
        self.beta = self.params.beta
        self.std_scale = self.params.std_scale
        self.lr = self.params.lr
        self.actor_loss = 0
        self.critic_loss = 0
        self.entropy_loss = 0

    def clear_memory_buffer(self):
        self.memory.clear()

    def add_last_state(self, last_state):
        self.memory.add_last_state(last_state)
    
    def step(self, states, actions, rewards, log_probs, values, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        # Save experience 
        self.memory.add(states, actions, rewards, log_probs, values, dones)

    def act(self, state, std_scale):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.ppo_ac_net.eval()
        with torch.no_grad():
            action, log_prob, entropy, value = self.ppo_ac_net(state, std_scale=std_scale)  # stdev cannot = 0
        self.ppo_ac_net.train()
        
        # Detach everything to ensure no backprop to these old experiences stored
        action = np.clip(action.cpu().detach().squeeze().numpy(), -1, 1)  
        log_prob = log_prob.cpu().detach().squeeze().numpy()  
        entropy = entropy.cpu().detach().squeeze().numpy()  
        value = value.cpu().detach().squeeze().numpy()  

        return action, log_prob, entropy, value

    def learn(self):
        """
        PPO-A3C Learning.
        """
        states, actions, rewards, log_probs, values, dones = self.memory.sample()

        # Append last value-state fn to values for GAE value computation
        if self.params.use_gae:
            _, _, _, last_values = self.act(self.memory.retrieve_last_state(), self.std_scale)
            last_values = np.expand_dims(last_values, axis=0)
            values = np.concatenate((values, last_values), axis=0)

        # Convert to Pytorch Tensors
        states = torch.from_numpy(states).float().to(self.params.device)
        actions = torch.from_numpy(actions).float().to(self.params.device).squeeze(1)
        log_probs = torch.from_numpy(log_probs).float().to(self.params.device).squeeze(1)
        values = torch.from_numpy(values).float().to(self.params.device)

        # Compute advantage for all robots
        advantages = []
        rewards_future = []
        values = values.cpu().numpy()
        returns = values[-1]   #(N, )
        advantage = 0
        for i in reversed(range(len(states))):  #(E, )
            returns = rewards[i] + self.params.gamma * (1-dones[i]) * returns
            if self.params.use_gae:
                td_err = rewards[i] + self.params.gamma * (1-dones[i]) * values[i+1] - values[i]
                advantage = advantage * self.params.gae_tau * self.params.gamma * (1-dones[i]) + td_err
            else:
                advantage = returns - values[i]
            advantages.append(advantage)
            rewards_future.append(returns)

        advantages = np.stack(advantages[::-1])            # Flip order (E, N)
        rewards_future = np.stack(rewards_future[::-1])    # Flip order (E, N)
        rewards_future = torch.tensor(rewards_future.copy()).float().to(self.params.device).detach()
        advantages_normalized = (advantages - np.nanmean(advantages)) / (np.std(advantages) + 1e-10)  # +1e-10
        advantages_normalized = torch.tensor(advantages_normalized.copy()).float().to(self.params.device).detach()

        # Flatten all components of experience into (E*N, xxx)
        num_exp = states.shape[0] * states.shape[1]
        states = states.view(num_exp, -1)                               # (N*E, S=129)
        actions = actions.view(num_exp, -1)                             # (N*E, A=20)
        log_probs = log_probs.view(num_exp)                             # (N*E,)
        rewards_future = rewards_future.view(num_exp)                   # (N*E,)
        advantages_normalized = advantages_normalized.view(num_exp)     # (N*E,)

        # Sample (traj_length/batch_size) batches of indices (of size=batch_size)
        # NOTE: These indices sets cover the entire range of indices
        batches = BatchSampler( SubsetRandomSampler(range(len(states))), self.params.batch_size, drop_last=False)
        actor_losses = []
        critic_losses = []
        entropy_losses = []

        for batch_idx in batches:
            
            # Filter out sampled data
            #batch_idx = torch.tensor(batch_idx)       # ADDED
            sampled_states = states[batch_idx]
            sampled_actions = actions[batch_idx]
            sampled_rewards = rewards_future[batch_idx]
            sampled_log_probs = log_probs[batch_idx]
            sampled_advantages = advantages_normalized[batch_idx]

            # Policy Loss (PPO)
            _, cur_log_probs, cur_ent, cur_values = self.ppo_ac_net(sampled_states, action=sampled_actions, std_scale=self.std_scale)
            ppo_ratio = (cur_log_probs - sampled_log_probs).exp()
            clip = torch.clamp(ppo_ratio, 1 - self.eps,  1 + self.eps)
            policy_loss = torch.min(ppo_ratio * sampled_advantages, clip * sampled_advantages)
            policy_loss = -torch.mean(policy_loss)

            # Critic Loss (MSE)
            critic_loss = F.mse_loss(sampled_rewards, cur_values.squeeze()) * self.params.critic_loss_coeff

            # Entropy Loss
            entropy_loss = -torch.mean(cur_ent) * self.beta

            # Compute Overall Loss 
            # NOTE: Maximize entropy to encourage exploration (beta to decay exploration with time)
            loss = policy_loss + critic_loss + entropy_loss 

            # Perform gradient ascent
            self.optimizer.zero_grad()
            loss.backward()
            if self.params.gradient_clip != 0:
                torch.nn.utils.clip_grad_norm_(self.ppo_ac_net.actor.parameters(), self.params.gradient_clip)    # To prevent exploding grad issue
                torch.nn.utils.clip_grad_norm_(self.ppo_ac_net.critic.parameters(), self.params.gradient_clip)   # To prevent exploding grad issue
            self.optimizer.step()

            # Post-processing
            actor_losses.append(policy_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy_loss.item())

        # Decay-able parameters
        self.eps = max(self.eps * self.params.eps_decay, self.params.eps_min)
        self.beta = max(self.beta * self.params.beta_decay, self.params.beta_min)
        self.std_scale = max(self.std_scale * self.params.std_scale_decay, self.params.std_scale_min)
        self.lr = max(self.lr * self.params.lr_decay, self.params.lr_min)
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

        # Store Stats
        self.actor_loss = sum(actor_losses) / len(actor_losses)
        self.critic_loss = sum(critic_losses) / len(critic_losses) 
        self.entropy_loss = sum(entropy_losses) / len(entropy_losses) 
        self.clear_memory_buffer()

   
    def print_init_messages(self):

        if self.params.verbose:
            print("\n=============== NETWORKS ===============")
            print("ppo_actor_critic_net", self.ppo_ac_net)
        
        print("\n~~~~~~ TRAINING ~~~~~")
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
    
    def __init__(self, state_size, total_state_size, action_size, params=Params()):
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
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []

    def clear_memory_buffer(self):
        self.memory.clear()

    def add_last_state(self, last_state):
        self.memory.add_last_state(last_state)
    
    def step(self, states, actions, rewards, log_probs, values, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        # Save experience 
        self.memory.add(states, actions, rewards, log_probs, values, dones)


    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.ppo_ac_net.eval()
        with torch.no_grad():
            action, log_prob, entropy, value = self.ppo_ac_net(state, std_scale=0.0)
        self.ppo_ac_net.train()
        
        action = np.clip(action.cpu().detach().numpy().item(), -1, 1)
        log_prob = log_prob.cpu().detach().numpy().item()  
        entropy = entropy.cpu().detach().numpy().item()  
        value = value.cpu().detach().numpy().item()  

        return action, log_prob, entropy, value

    def learn(self):
        """
        PPO-A3C Learning.
        """
        states, actions, rewards, log_probs, values, dones = self.memory.sample()
    
        # Append last value-state fn to values for GAE value computation
        if self.params.use_gae:
            last_state = np.expand_dims(self.memory.retrieve_last_state(), axis=0)
            _, _, _, value = self.ppo_ac_net(last_state)
            values = np.append(values, value, axis=0)

        states = torch.from_numpy(states).float().to(self.params.device)
        actions = torch.from_numpy(actions).long().to(self.params.device).squeeze(1)
        log_probs = torch.from_numpy(log_probs).float().to(self.params.device).squeeze(1)
        values = torch.from_numpy(values).float().to(self.params.device)

        # if self.params.use_gae:
        advantages = []
        rewards_future = []
        values = values.cpu().numpy()
        returns = values[-1]
        advantage = 0
        for i in reversed(range(len(states))):
            returns = rewards[i] + self.params.gamma * (1-dones[i]) * returns
            if self.params.use_gae:
                td_err = rewards[i] + self.params.gamma * (1-dones[i]) * values[i+1] - values[i]
                advantage = advantage * self.params.gae_tau * self.params.gamma * (1-dones[i]) + td_err
            else:
                advantage = returns - values[i]
            advantages.append(float(advantage))
            rewards_future.append(float(returns))
        
        advantages = advantages[::-1]            # Flip order
        rewards_future = rewards_future[::-1]    # Flip order
        rewards_future = torch.tensor(rewards_future.copy()).float().to(self.params.device).detach()
        advantages = torch.tensor(advantages.copy()).float().to(self.params.device).detach()
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        # Sample (traj_length/batch_size) batches of indices (of size=batch_size)
        # NOTE: These indices sets cover the entire range of indices
        batches = BatchSampler( SubsetRandomSampler(range(len(states))), self.params.batch_size, drop_last=False)
        self.actor_losses.clear()
        self.critic_losses.clear()
        self.entropy_losses.clear()

        for batch_idx in batches:
            
            # Filter out sampled data
            batch_idx = torch.tensor(batch_idx).long().to(self.params.device)       # ADDED
            sampled_states = states[batch_idx]
            sampled_actions = actions[batch_idx]
            sampled_rewards = rewards_future[batch_idx]
            sampled_log_probs = log_probs[batch_idx]
            sampled_advantages = advantages_normalized[batch_idx]

            # Policy Loss (PPO)
            _, cur_log_probs, entropies, values = self.ppo_ac_net(sampled_states, action=sampled_actions, std_scale=self.std_scale)
            ppo_ratio = (cur_log_probs - sampled_log_probs).exp()
            clip = torch.clamp(ppo_ratio, 1 - self.eps,  1 + self.eps)
            policy_loss = torch.min(ppo_ratio * sampled_advantages, clip * sampled_advantages)
            policy_loss = -torch.mean(policy_loss)

            # Critic Loss (MSE)
            critic_loss = F.mse_loss(sampled_rewards, values.squeeze(1))

            # Entropy Loss
            entropy_loss = -torch.mean(entropies) * self.beta

            # Compute Overall Loss 
            # NOTE: Maximize entropy to encourage exploration (beta to decay exploration with time)
            loss = policy_loss + (0.5 * critic_loss) + entropy_loss 

            # Perform gradient ascent
            self.optimizer.zero_grad()
            loss.backward()
            if self.params.gradient_clip != 0:
                torch.nn.utils.clip_grad_norm_(self.ppo_ac_net.parameters(), self.params.gradient_clip)    # ADDED: Gradient Clipping to prevent exploding grad issue
            self.optimizer.step()

            # Post-processing
            self.actor_losses.append(policy_loss.item())
            self.critic_losses.append(critic_loss.item())
            self.entropy_losses.append(entropy_loss.item())

        # self.eps *= self.params.eps_decay
        # self.beta *= self.params.beta_decay
        self.eps = max(self.eps * self.params.eps_decay, self.params.eps_min)
        self.beta = max(self.beta * self.params.beta_decay, self.params.beta_min)
        self.std_scale = max(self.beta * self.params.std_scale_decay, self.params.std_scale_min)
        self.clear_memory_buffer()

   
    def print_init_messages(self):

        if self.params.verbose:
            print("\n=============== NETWORKS ===============")
            print("ppo_actor_critic_net", self.ppo_ac_net)
        
        print("\n~~~~~~ TRAINING ~~~~~")
import numpy as np
import random
import copy
from collections import namedtuple, deque
from model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim
from params import Params
from replay_buffer import ReplayBuffer
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class PPO_Agent():
    """
    PyTorch Implementation of Proximal Policy Optimization (PPO) with A3C Value Estimation:
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
        self.actor_net = Actor(state_size, action_size, self.params).to(self.device)
        self.critic_net = Critic(total_state_size, action_size, self.params).to(self.device)
        self.optimizer = optim.Adam(self.critic_net.parameters(), lr=self.params.lr_critic, weight_decay=self.params.weight_decay, eps=self.params.optimizer_eps)

        # Others
        self.memory = ReplayBuffer(state_size, action_size, self.params.random_seed, params)
        self.t_step = 0
        self.learn_step = 0
        self.actor_loss = 0
        self.critic_loss = 0

    def clear_memory_buffer(self):
        self.memory.clear()

    def step(self, states, all_states, actions, rewards, log_probs):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        # Save experience 
        log_probs = log_probs.cpu().numpy()
        self.memory.add(states, all_states, actions, rewards, log_probs)


    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_net.eval()
        with torch.no_grad():
            action, log_prob, entropy = self.actor_net(state)
            action = action.cpu().data            
        self.actor_net.train()
        
        return action, log_prob, entropy

    def learn(self):
        """
        PPO-A3C Learning.
        """
        states, all_states, actions, rewards, log_probs = self.memory.sample()

        # convert rewards to future rewards 
        rewards = rewards.cpu().numpy()
        discount = self.params.gamma**np.arange(len(rewards))
        rewards = np.asarray(rewards)*discount[:,np.newaxis]
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1] 
    
        # normalize advantage function
        self.critic_net.eval()
        with torch.no_grad():
            values = self.critic_net(all_states).cpu().detach().numpy()
        self.critic_net.train()
        advantages = (rewards_future - values)
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        advantages_normalized = torch.tensor(advantages_normalized).float().to(self.params.device)
        rewards = torch.from_numpy(rewards).to(self.params.device).type(torch.float)

        # Sample (traj_length/batch_size) batches of indices (of size=batch_size)
        # NOTE: These indices sets cover the entire range of indices
        batches = BatchSampler( SubsetRandomSampler(range(len(states))), self.params.batch_size, drop_last=False)

        for batch_idx in batches:
            
            # Filter out sampled data
            sampled_states = states[batch_idx]
            sampled_all_states = all_states[batch_idx]
            sampled_actions = actions[batch_idx]
            sampled_rewards = rewards[batch_idx]
            sampled_log_probs = log_probs[batch_idx]
            sampled_advantages = advantages_normalized[batch_idx]

            # Policy Loss (PPO)
            _, cur_log_probs, entropies = self.actor_net(sampled_states, sampled_actions)
            ppo_ratio = (cur_log_probs.unsqueeze(1) - sampled_log_probs).exp()
            clip = torch.clamp(ppo_ratio, 1 - self.params.eps,  1 + self.params.eps)
            policy_loss = -torch.min(ppo_ratio * sampled_advantages, clip * sampled_advantages).mean()

            # Critic Loss (MSE)
            sampled_values = self.critic_net(sampled_all_states)
            critic_loss = F.mse_loss(sampled_rewards, sampled_values)

            # Compute Overall Loss 
            # NOTE: Maximize entropy to encourage exploration (beta to decay exploration with time)
            loss = policy_loss + (0.5 * critic_loss) - (entropies.mean() * self.params.beta)  

            # Perform gradient ascent
            self.optimizer.zero_grad()
            loss.backward()
            if self.params.gradient_clip != 0:
                torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.params.gradient_clip)    # ADDED: Gradient Clipping to prevent exploding grad issue
                torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.params.gradient_clip)   # ADDED: Gradient Clipping to prevent exploding grad issue
            self.optimizer.step()

        # Post-processing
        self.actor_loss = policy_loss.item()
        self.critic_loss = critic_loss.item()
        self.params.eps *= self.params.eps_decay
        self.params.beta *= self.params.beta_decay
        self.clear_memory_buffer()

   
    def print_init_messages(self):

        if self.params.verbose:
            print("\n=============== NETWORKS ===============")
            print("actor_net", self.actor_net)
            print("critic_net", self.critic_net)
        
        self.params.print_init_messages()
        print("\n~~~~~~ TRAINING ~~~~~")
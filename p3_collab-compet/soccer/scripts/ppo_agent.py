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

    The actor receives his own state space and outputs an action, the log probability of that action 
    (to be used later in calculating the advantage ratio) and the entropy of the probability distribution. 
    Higher entropy indicates more uncertainty in the probabilities. For example, when the probability of the 
    goalie choosing 1 of the 4 possible actions is roughly equal (25% each), this would be maximum entropy. 
    When one of those actions has 100% probability and the other 3 actions have 0% probability, the agent is absolutely 
    certain and entropy will be zero. 

    We use entropy as noise in the loss function to force the agent to try more random actions early on and not get fixated 
    on a solution which is not optimal in the long run (stuck in a local minima.). Since we are performing gradient descent on 
    the negative of entropy, we are maximizing it. However, the decaying beta value will continuously shrink the contribution 
    of entropy in the loss function, leading to more optimization to minimize policy and value loss. Hence, we will notice a 
    dip in entropy with entropy with time as the agent's policy and critic nets becomes increasingly confident of their predictions.

    The critic receives the combined state space of all 4 agents on the field and outputs the expected average value 
    (total reward) for an action taken given that state. It learns in a supervised learning fashion by optimizing the MSE loss
    between future cumulative reward vs state-value estimation. State-value estimates converges with sufficient exploration. 

    The advantage function is used in computing policy loss to indicate how much better an agent is performing relative to a baseline. 
    This baseline is the state-value prediction from the critic network on how much rewards an agent ought to receive given a state. 
    Hence, as an agent improves (make better actions and more accurately predict value of states), it is forced to make even better
    actions that yield higher rewards than what is thought to be the 'averaged' reward for being in that particular state. In simple 
    terms, an R=+30 may be good at the start, but not as desirable in later training phases.

    A note on the distributions function:
    It is not possible to have the actor simply output a softmax distribution of action probabilities and then choose an action 
    off a random sampling of those probabilities. Neural networks cannot directly backpropagate through random samples. 
    PyTorch and Tensorflow offer a distribution function to solve this that makes the action selection differentiable. 
    The actor passes the softmax output through this distribution function to select the action and then backpropagation can occur.
    https://pytorch.org/docs/stable/distributions.html

    - Adapted from Paul Hrabal
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
        # NOTE: Optimizer jointly optimizes actor and critic through joint loss function
        self.actor_net = Actor(state_size, action_size, self.params).to(self.device)
        self.critic_net = Critic(total_state_size, self.params).to(self.device)
        self.optimizer = optim.Adam(list(self.actor_net.parameters()) + list(self.critic_net.parameters()), lr=self.params.lr)

        # Others
        self.memory = ReplayBuffer(state_size, action_size, self.params.random_seed, params)
        self.eps = self.params.eps
        self.beta = self.params.beta
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []

    def clear_memory_buffer(self):
        self.memory.clear()

    def step(self, states, all_states, actions, rewards, log_probs):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        # Save experience 
        #log_probs = log_probs.cpu().numpy()
        self.memory.add(states, all_states, actions, rewards, log_probs)


    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor_net.eval()
        with torch.no_grad():
            action, log_prob, _ = self.actor_net(state)
        self.actor_net.train()
        
        action = action.cpu().detach().numpy().item()
        log_prob = log_prob.cpu().detach().numpy().item()  

        return action, log_prob

    def learn(self):
        """
        PPO-A3C Learning.
        """
        states, all_states, actions, rewards, log_probs = self.memory.sample()

        discount = self.params.gamma**np.arange(len(rewards))
        rewards = rewards.squeeze(1) * discount
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
        states = torch.from_numpy(states).float().to(self.params.device)
        all_states = torch.from_numpy(all_states).float().to(self.params.device)
        actions = torch.from_numpy(actions).long().to(self.params.device).squeeze(1)
        log_probs = torch.from_numpy(log_probs).float().to(self.params.device).squeeze(1)
        rewards_future = torch.from_numpy(rewards_future.copy()).float().to(self.params.device)


        # normalize advantage function
        self.critic_net.eval()
        with torch.no_grad():
            values = self.critic_net(all_states).detach()
        self.critic_net.train()

        advantages = (rewards_future - values.squeeze(1)).detach()
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        # advantages_normalized = torch.tensor(advantages_normalized).float().to(self.params.device).detach()
        # rewards_future = torch.from_numpy(np.array(rewards_future)).to(self.params.device).type(torch.float).detach()

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
            sampled_all_states = all_states[batch_idx]
            sampled_actions = actions[batch_idx]
            sampled_rewards = rewards_future[batch_idx]
            sampled_log_probs = log_probs[batch_idx]
            sampled_advantages = advantages_normalized[batch_idx]

            # Policy Loss (PPO)
            _, cur_log_probs, entropies = self.actor_net(sampled_states, sampled_actions)
            ppo_ratio = (cur_log_probs - sampled_log_probs).exp()
            clip = torch.clamp(ppo_ratio, 1 - self.eps,  1 + self.eps)
            policy_loss = torch.min(ppo_ratio * sampled_advantages, clip * sampled_advantages)
            policy_loss = -torch.mean(policy_loss)

            # Critic Loss (MSE)
            sampled_values = self.critic_net(sampled_all_states)
            critic_loss = F.mse_loss(sampled_rewards, sampled_values.squeeze(1))

            # Entropy Loss
            entropy_loss = torch.mean(entropies)

            # Compute Overall Loss 
            # NOTE: Maximize entropy to encourage exploration (beta to decay exploration with time)
            loss = policy_loss + (0.5 * critic_loss) - (entropy_loss * self.beta)  

            # Perform gradient ascent
            self.optimizer.zero_grad()
            loss.backward()
            if self.params.gradient_clip != 0:
                torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.params.gradient_clip)    # ADDED: Gradient Clipping to prevent exploding grad issue
                torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.params.gradient_clip)   # ADDED: Gradient Clipping to prevent exploding grad issue
            self.optimizer.step()

            # Post-processing
            self.actor_losses.append(policy_loss.item())
            self.critic_losses.append(critic_loss.item())
            self.entropy_losses.append(entropy_loss.item())

        # self.eps *= self.params.eps_decay
        # self.beta *= self.params.beta_decay
        self.eps = max(self.eps * self.params.eps_decay, self.params.eps_min)
        self.beta = max(self.beta * self.params.beta_decay, self.params.beta_min)
        self.clear_memory_buffer()

   
    def print_init_messages(self):

        if self.params.verbose:
            print("\n=============== NETWORKS ===============")
            print("actor_net", self.actor_net)
            print("critic_net", self.critic_net)
        
        print("\n~~~~~~ TRAINING ~~~~~")
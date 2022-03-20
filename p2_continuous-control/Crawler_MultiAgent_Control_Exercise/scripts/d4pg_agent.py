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


class D4PG_Agent():
    """
    PyTorch Implementation of D4PG:
    "Distributed Distributional Deterministic Policy Gradients"
    (Barth-Maron, Hoffman, et al., 2018)
    As described in the paper at: https://arxiv.org/pdf/1804.08617.pdf

    Much thanks also to the original DDPG paper:
    "Continuous Control with Deep Reinforcement Learning"
    (Lillicrap, Hunt, et al., 2016)
    https://arxiv.org/pdf/1509.02971.pdf

    And to:
    "A Distributional Perspective on Reinforcement Learning"
    (Bellemare, Dabney, et al., 2017)
    https://arxiv.org/pdf/1707.06887.pdf

    D4PG utilizes distributional value estimation, n-step returns,
    prioritized experience replay (PER), distributed K-actor exploration,
    and off-policy actor-critic learning to achieve very fast and stable
    learning for continuous control tasks.

    This version of the Agent is written to interact with Udacity's
    Continuous Control robotic arm manipulation environment which provides
    20 simultaneous actors, negating the need for K-actor implementation.
    Thus, this code has no multiprocessing functionality. It could be easily
    added as part of the main.py script.

    In the original D4PG paper, it is suggested in the data that PER does
    not have significant (or perhaps any at all) effect on the speed or
    stability of learning. Thus, it too has been left out of this
    implementation but may be added as a future TODO item.

    Credits: 
        Zhang Shangtong: https://github.com/ShangtongZhang
        whiterabbitobj: https://github.com/whiterabbitobj
    """
    
    def __init__(self, state_size, action_size, params=Params()):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """

        # torch.autograd.set_detect_anomaly(True)         

        self.params = params
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(self.params.random_seed)
        self.device = self.params.device

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, self.params).to(self.device)
        self.actor_target = Actor(state_size, action_size, self.params).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.params.lr_actor, weight_decay=self.params.weight_decay, eps=self.params.optimizer_eps)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, self.params).to(self.device)
        self.critic_target = Critic(state_size, action_size, self.params).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.params.lr_critic, weight_decay=self.params.weight_decay, eps=self.params.optimizer_eps)

        # Others
        self.noise = OUNoise(action_size, self.params.random_seed, theta=self.params.action_noise_theta, sigma=self.params.action_noise_sigma)
        self.memory = ReplayBuffer(action_size, self.params.buffer_size, self.params.batch_size, self.params.random_seed, params)
        self.t_step = 0
        self.learn_step = 0
        self.memory_prefilled_alerted = False
        self.atoms = torch.linspace(self.params.vmin, self.params.vmax, self.params.num_atoms).to(self.device)
        self.predicted_probs = np.zeros((self.params.batch_size, self.params.num_atoms))
        self.projected_target_probs = np.zeros((self.params.batch_size, self.params.num_atoms))

        # Outputs hyperparams
        self.print_init_messages(params)
    
    def memory_buffer_prefilled(self):
        return len(self.memory) >= self.params.prefill_memory_qty

    def step(self, states, actions, rewards, next_states, dones, clear_nstep_buffer=False):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        # Save experience (let ReplayBuffer class handle N-Step Bootstrapping)
        cur_experiences = (states, actions, rewards, next_states, dones)
        self.memory.fill_nstep_buffer(cur_experiences, clear_nstep_buffer)

        if len(self.memory) < self.params.prefill_memory_qty:
            print("\rPrefilling Replay Memory Buffer: {} / {}".format(len(self.memory), self.params.prefill_memory_qty), end="")

        # Learn every UPDATE_EVERY time steps.
        else:
            if not self.memory_prefilled_alerted: 
                print("\rPrefilling Replay Memory Buffer: {} / {}".format(len(self.memory), self.params.prefill_memory_qty), end="")
                print("\n ===== Replay Buffer prefilled. Agent will begin learning. =====\n")
                self.memory_prefilled_alerted = True
            
            self.t_step = (self.t_step + 1) % self.params.learn_every      
            if self.t_step == 0 and len(self.memory) > self.params.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.params.gamma)
                
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
            # action += (0.3 * (np.random.normal(0, 1, action.shape)))   // SUGGESTED IN D4PG PAPER
                
        #return np.clip(action, -1, 1)
        return action

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """
        Performs a distributional Actor/Critic calculation and update.
        Actor πθ and πθ'
        Critic Zw and Zw' (categorical distribution)
        """

        states, actions, rewards, next_states, dones = experiences

        # Sample from replay buffer, REWARDS are sum of ROLLOUT timesteps
        # Already calculated before storing in the replay buffer.
        # NEXT_STATES are ROLLOUT steps ahead of STATES
        atoms = self.atoms.unsqueeze(0)
        # Calculate Yᵢ from target networks using πθ' and Zw'
        # These tensors are not needed for backpropogation, so detach from the
        # calculation graph (literally doubles runtime if this is not detached)
        target_actions = self.actor_target(next_states)
        target_probs = self.critic_target(next_states, target_actions).detach()
        projected_target_probs = self.categorical_projection(rewards, target_probs, dones)

        # Calculate log probability DISTRIBUTION using Zw w.r.t. stored actions
        log_probs = self.critic_local(states, actions, log=True)

        # Storage for visualization later
        self.projected_target_probs = projected_target_probs[0].clone().cpu().detach()
        self.predicted_probs = log_probs[0].clone().exp().cpu().detach()  # STORE TO DEBUG

        # Calculate the critic network LOSS (Cross Entropy), CE-loss is ideal
        # for categorical value distributions as utilized in D4PG.
        # estimates distance between target and projected values
        critic_loss = -(projected_target_probs * log_probs).sum(-1).mean()

        # Predict action for actor network loss calculation using πθ
        predicted_action = self.actor_local(states)

        # Predict value DISTRIBUTION using Zw w.r.t. action predicted by πθ
        probs = self.critic_local(states, predicted_action)

        # Multiply probabilities by atom values and sum across columns to get
        # Q-Value
        expected_reward = (probs * atoms).sum(-1)

        # Calculate the actor network LOSS (Policy Gradient)
        # Take the mean across the batch and multiply in the negative to
        # perform gradient ascent
        actor_loss = -expected_reward.mean()

        # Perform gradient ascent
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.params.gradient_clip != 0:
            torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), self.params.gradient_clip)    # ADDED: Gradient Clipping to prevent exploding grad issue
        self.actor_optimizer.step()

        # Perform gradient descent
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.params.gradient_clip != 0:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.params.gradient_clip)    # ADDED: Gradient Clipping to prevent exploding grad issue
        self.critic_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        # Learn every UPDATE_EVERY time steps.
        if self.params.hard_update:
            self.learn_step = (self.learn_step + 1) % self.params.hard_weights_update_every
        else:
            self.learn_step = (self.learn_step + 1) % self.params.soft_weights_update_every
        
        if self.params.hard_update and self.learn_step == 0:
            self.hard_update(self.critic_local, self.critic_target)
            self.hard_update(self.actor_local, self.actor_target)  

        elif not self.params.hard_update and self.learn_step == 0:
            self.soft_update(self.critic_local, self.critic_target, self.params.tau)
            self.soft_update(self.actor_local, self.actor_target, self.params.tau)   


        self.actor_loss = actor_loss.item()
        self.critic_loss = critic_loss.item()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        """
        Fully copy parameters from active network to target network. To be used
        in conjunction with a parameter "C" that modulated how many timesteps
        between these hard updates.
        """

        target_model.load_state_dict(local_model.state_dict())

    # def get_projected_critic_target(self, rewards, next_states):
    #     """
    #     Calculate Yᵢ from target networks using πθ' and Zw'
    #     """

    #     target_actions = self.actor_target(next_states)
    #     target_probs = self.critic_target(next_states, target_actions)
    #     # Project the categorical distribution onto the supports
    #     projected_probs = self.categorical_projection(rewards, target_probs)
    #     return projected_probs

    def categorical_projection(self, rewards, probs, dones):
        """
        Returns the projected value distribution for the input state/action pair

        While there are several very similar implementations of this Categorical
        Projection methodology around github, this is one of them.
        """

        # Create local vars to keep code more concise
        vmin = self.params.vmin
        vmax = self.params.vmax
        atoms = self.atoms
        num_atoms = self.params.num_atoms
        gamma = self.params.gamma
        rollout = self.params.n_step_bootstrap

        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1).type(torch.float)
        delta_z = (vmax - vmin) / (num_atoms - 1)

        # Rewards were stored with 0->(N-1) summed, take Reward and add it to
        # the discounted expected reward at N (ROLLOUT) timesteps
        projected_atoms = rewards + gamma**rollout * atoms.unsqueeze(0) * (1 - dones)
        projected_atoms.clamp_(vmin, vmax)
        b = ((projected_atoms - vmin) / delta_z).squeeze()

        # It seems that on professional level GPUs (for instance on AWS), the
        # floating point math is accurate to the degree that a tensor printing
        # as 99.00000 might in fact be 99.000000001 in the backend, perhaps due
        # to binary imprecision, but resulting in 99.00000...ceil() evaluating
        # to 100 instead of 99. Forcibly reducing the precision to the minimum
        # seems to be the only solution to this problem, and presents no issues
        # to the accuracy of calculating lower/upper_bound correctly.
        precision = 1
        b = torch.round(b * 10**precision) / 10**precision
        lower_bound = b.floor()
        upper_bound = b.ceil()

        # Don't think (lower_bound == upper_bound) matters here
        m_lower = (upper_bound + (lower_bound == upper_bound).float() - b) * probs
        m_upper = (b - lower_bound) * probs

        projected_probs = torch.tensor(np.zeros(probs.size())).to(self.device)

        for idx in range(probs.size(0)):
            projected_probs[idx].index_add_(0, lower_bound[idx].long(), m_lower[idx].double())
            projected_probs[idx].index_add_(0, upper_bound[idx].long(), m_upper[idx].double())
        return projected_probs.float()


    def print_init_messages(self, params):

        if self.params.verbose:
            print("\n=============== NETWORKS ===============")
            print("actor_local", self.actor_local)
            print("actor_target", self.actor_target)
            print("critic_local", self.critic_local)
            print("critic_target", self.critic_target)
        
        self.params.print_init_messages()

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta*(self.mu - x) + self.sigma*np.random.standard_normal(len(x))   # The more proper way to simulate noise
        self.state = x + dx
        return self.state
import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(3e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 4e-4        # learning rate of the critic
WEIGHT_DECAY = 1e-2     # L2 weight decay          (ORIGINAL: 0)
HARD_UPDATE=True        # Hard update OR Soft Update?
LEARN_EVERY = 1                  # how often for local networks to learn
SOFT_WEIGHTS_UPDATE_EVERY = 20   # how often to copy weights over to target networks (Gradually)
HARD_WEIGHTS_UPDATE_EVERY = 350  # how often to copy weights over to target networks (Instant)
N_STEP_BOOTSTRAP = 5             # N-Step bootstrapping for Temporal Difference Update Calculations

# D4PG STUFF
# Lower and upper bounds of critic value output distribution, these will vary with environment
# V_min and V_max should be chosen based on the range of normalised reward values in the chosen env
# Assume Normalized Reward = Rewards / 100
VMIN = 0
VMAX = 0.3
NUM_ATOMS = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ATOMS = torch.linspace(VMIN, VMAX, NUM_ATOMS).to(device)

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, NUM_ATOMS).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, NUM_ATOMS).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.t_step = 0
        self.learn_step = 0
        self.memory_prefilled_alerted = False
        
        ## Print networks
        print("=============== NETWORKS ===============")
        print("actor_local", self.actor_local)
        print("actor_target", self.actor_target)
        print("critic_local", self.critic_local)
        print("critic_target", self.critic_target)
        
        # Print Hyper-parameters
        print("\n=============== HYPERPARAMS ===============")
        print("DEVICE: ", device)
        print("BUFFER_SIZE: ", BUFFER_SIZE)
        print("BATCH_SIZE: ", BATCH_SIZE)
        print("GAMMA: ", GAMMA)
        print("TAU: ", TAU)
        print("LR_ACTOR: ", LR_ACTOR)
        print("LR_CRITIC: ", LR_CRITIC)
        print("WEIGHT_DECAY: ", WEIGHT_DECAY)
        print("HARD_UPDATE: ", HARD_UPDATE)
        print("LEARN_EVERY: ", LEARN_EVERY)
        if (HARD_UPDATE):
            print("HARD_WEIGHTS_UPDATE_EVERY: ", HARD_WEIGHTS_UPDATE_EVERY)
        else:
            print("SOFT_WEIGHTS_UPDATE_EVERY: ", SOFT_WEIGHTS_UPDATE_EVERY)
        print("N_STEP_BOOTSTRAP: ", N_STEP_BOOTSTRAP)
        print("VMIN: ", VMIN)
        print("VMAX: ", VMAX)
        print("NUM_ATOMS: ", NUM_ATOMS)
        print("===========================================\n")

    
    def memory_buffer_prefilled(self, prefill_memory_qty):
        return len(self.memory) >= prefill_memory_qty

    def step(self, state, action, reward, next_state, done, prefill_memory_qty=0):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) < prefill_memory_qty:
            print("\rPrefilling Replay Memory Buffer: {} / {}".format(len(self.memory), prefill_memory_qty), end="")

        # Learn every UPDATE_EVERY time steps.
        else:
            
            if not self.memory_prefilled_alerted: 
                print("\rPrefilling Replay Memory Buffer: {} / {}".format(len(self.memory), prefill_memory_qty), end="")
                print("\n ===== Replay Buffer prefilled. Agent will begin learning. =====\n")
                self.memory_prefilled_alerted = True
            
            self.t_step = (self.t_step + 1) % LEARN_EVERY        
            if self.t_step == 0:
                # Learn, if enough samples are available in memory
                if len(self.memory) > BATCH_SIZE:
    #                 print("LEARNING, self.t_step: ", self.t_step)
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)
                    # print("LEARNING...")
                
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
                
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

                # Sample from replay buffer, REWARDS are sum of ROLLOUT timesteps
        # Already calculated before storing in the replay buffer.
        # NEXT_STATES are ROLLOUT steps ahead of STATES
        atoms = ATOMS.unsqueeze(0)
        # Calculate Yᵢ from target networks using πθ' and Zw'
        # These tensors are not needed for backpropogation, so detach from the
        # calculation graph (literally doubles runtime if this is not detached)
        target_dist = self._get_targets(rewards, next_states).detach()

        # Calculate log probability DISTRIBUTION using Zw w.r.t. stored actions
        log_probs = self.critic_local(states, actions, log=True)

        # Calculate the critic network LOSS (Cross Entropy), CE-loss is ideal
        # for categorical value distributions as utilized in D4PG.
        # estimates distance between target and projected values
        critic_loss = -(target_dist * log_probs).sum(-1).mean()


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
        self.actor_optimizer.step()

        # Perform gradient descent
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # self._update_networks()

        # self.actor_loss = actor_loss.item()
        # self.critic_loss = critic_loss.item()

        # # ---------------------------- update critic ---------------------------- #
        # # Get predicted next-state actions and Q values from target models
        # actions_next = self.actor_target(next_states)
        # Q_targets_next = self.critic_target(next_states, actions_next)

        # # Compute Q targets for current states (y_i)
        # # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Q_targets = rewards + ((gamma**N_STEP_BOOTSTRAP) * Q_targets_next * (1 - dones))

        # # Compute critic loss
        # Q_expected = self.critic_local(states, actions)
        # critic_loss = F.mse_loss(Q_expected, Q_targets)
        # # Minimize the loss
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()

        # # ---------------------------- update actor ---------------------------- #
        # # Compute actor loss
        # actions_pred = self.actor_local(states)
        # actor_loss = -self.critic_local(states, actions_pred).mean()
        # # Minimize the loss
        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        # Learn every UPDATE_EVERY time steps.
        self.learn_step = (self.learn_step + 1)   
#         print("SOFT COPYING WEIGHTS, self.learn_step: ", self.learn_step)
        
        if HARD_UPDATE and (self.learn_step % HARD_WEIGHTS_UPDATE_EVERY) == 0:
            self.hard_update(self.critic_local, self.critic_target)
            self.hard_update(self.actor_local, self.actor_target)  

        elif not HARD_UPDATE and (self.learn_step % SOFT_WEIGHTS_UPDATE_EVERY) == 0:
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)   

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

    ############################################# ADDED ####################################

    def _get_targets(self, rewards, next_states):
        """
        Calculate Yᵢ from target networks using πθ' and Zw'
        """

        target_actions = self.actor_target(next_states)
        target_probs = self.critic_target(next_states, target_actions)
        # Project the categorical distribution onto the supports
        projected_probs = self._categorical(rewards, target_probs)
        return projected_probs

    def _categorical(self, rewards, probs):
        """
        Returns the projected value distribution for the input state/action pair

        While there are several very similar implementations of this Categorical
        Projection methodology around github, this function owes the most
        inspiration to Zhang Shangtong and his excellent repository located at:
        https://github.com/ShangtongZhang
        """

        # Create local vars to keep code more concise
        vmin = VMIN
        vmax = VMAX
        atoms = ATOMS
        num_atoms = NUM_ATOMS
        gamma = GAMMA
        rollout = N_STEP_BOOTSTRAP

        rewards = rewards.unsqueeze(-1)
        delta_z = (vmax - vmin) / (num_atoms - 1)

        # Rewards were stored with 0->(N-1) summed, take Reward and add it to
        # the discounted expected reward at N (ROLLOUT) timesteps
        projected_atoms = rewards + gamma**rollout * atoms.unsqueeze(0)
        projected_atoms.clamp_(vmin, vmax)
        b = ((projected_atoms - vmin) / delta_z).squeeze()

        # print("b: ", b.squeeze().shape)

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

        m_lower = (upper_bound + (lower_bound == upper_bound).float() - b) * probs
        m_upper = (b - lower_bound) * probs

        projected_probs = torch.tensor(np.zeros(probs.size())).to(device)

        for idx in range(probs.size(0)):
            projected_probs[idx].index_add_(0, lower_bound[idx].long(), m_lower[idx].double())
            projected_probs[idx].index_add_(0, upper_bound[idx].long(), m_upper[idx].double())
        return projected_probs.float()

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

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
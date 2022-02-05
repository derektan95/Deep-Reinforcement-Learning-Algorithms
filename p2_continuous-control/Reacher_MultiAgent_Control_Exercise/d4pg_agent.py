import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim
from params import Params


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, params=Params()):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.params = params
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, random_seed).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.params.lr_actor, weight_decay=self.params.weight_decay)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, self.params.num_atoms).to(DEVICE)
        self.critic_target = Critic(state_size, action_size, random_seed, self.params.num_atoms).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.params.lr_critic, weight_decay=self.params.weight_decay)

        # Others
        self.noise = OUNoise(action_size, random_seed)
        self.memory = ReplayBuffer(action_size, self.params.buffer_size, self.params.batch_size, random_seed, params)
        self.t_step = 0
        self.learn_step = 0
        self.memory_prefilled_alerted = False
        self.atoms = torch.linspace(self.params.vmin, self.params.vmax, self.params.num_atoms).to(DEVICE)
        
        
        # ## Print networks
        # print("\n=============== NETWORKS ===============")
        # print("actor_local", self.actor_local)
        # print("actor_target", self.actor_target)
        # print("critic_local", self.critic_local)
        # print("critic_target", self.critic_target)
        
        # Print Hyper-parameters
        print("\n=============== HYPERPARAMS ===============")
        print("DEVICE: ", DEVICE)
        print("BUFFER_SIZE: ", self.params.buffer_size)
        print("BATCH_SIZE: ", self.params.batch_size)
        print("GAMMA: ", self.params.gamma)
        print("TAU: ", self.params.tau)
        print("LR_ACTOR: ", self.params.lr_actor)
        print("LR_CRITIC: ", self.params.lr_critic)
        print("WEIGHT_DECAY: ", self.params.weight_decay)
        print("HARD_UPDATE: ", self.params.hard_update)
        print("LEARN_EVERY: ", self.params.learn_every)
        if (self.params.hard_update):
            print("HARD_WEIGHTS_UPDATE_EVERY: ", self.params.hard_weights_update_every)
        else:
            print("SOFT_WEIGHTS_UPDATE_EVERY: ", self.params.soft_weights_update_every)
        print("N_STEP_BOOTSTRAP: ", self.params.n_step_bootstrap)
        print("VMIN: ", self.params.vmin)
        print("VMAX: ", self.params.vmax)
        print("NUM_ATOMS: ", self.params.num_atoms)
        print("===========================================\n")

    
    def memory_buffer_prefilled(self, prefill_memory_qty):
        return len(self.memory) >= prefill_memory_qty

    def step(self, states, actions, rewards, next_states, dones, prefill_memory_qty=0, clear_nstep_buffer=False):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        # Save experience (let ReplayBuffer class handle N-Step Bootstrapping)
        # cur_experiences = (states, actions, rewards, next_states, dones)
        cur_experiences = list(zip(states, actions, rewards, next_states, dones))
        self.memory.fill_nstep_buffer(cur_experiences, clear_nstep_buffer)

        if len(self.memory) < prefill_memory_qty:
            print("\rPrefilling Replay Memory Buffer: {} / {}".format(len(self.memory), prefill_memory_qty), end="")

        # Learn every UPDATE_EVERY time steps.
        else:
            
            if not self.memory_prefilled_alerted: 
                print("\rPrefilling Replay Memory Buffer: {} / {}".format(len(self.memory), prefill_memory_qty), end="")
                print("\n ===== Replay Buffer prefilled. Agent will begin learning. =====\n")
                self.memory_prefilled_alerted = True
            
            self.t_step = (self.t_step + 1) % self.params.learn_every      
            if self.t_step == 0:
                # Learn, if enough samples are available in memory
                if len(self.memory) > self.params.batch_size:
    #                 print("LEARNING, self.t_step: ", self.t_step)
                    experiences = self.memory.sample()
                    self.learn(experiences, self.params.gamma)
                
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            # action += self.noise.sample()
            action += (0.3 * (np.random.normal(0, 1, action.shape)))
                
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
        atoms = self.atoms.unsqueeze(0)
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

        # # ---------------------------- update critic ---------------------------- #
        # # Get predicted next-state actions and Q values from target models
        # actions_next = self.actor_target(next_states)
        # Q_targets_next = self.critic_target(next_states, actions_next)

        # # Compute Q targets for current states (y_i)
        # # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Q_targets = rewards + ((gamma**self.params.n_step_bootstrap) * Q_targets_next * (1 - dones))

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
        vmin = self.params.vmin
        vmax = self.params.vmax
        atoms = self.atoms
        num_atoms = self.params.num_atoms
        gamma = self.params.gamma
        rollout = self.params.n_step_bootstrap

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
        # m_lower = (upper_bound - b) * probs
        m_upper = (b - lower_bound) * probs

        projected_probs = torch.tensor(np.zeros(probs.size())).to(DEVICE)

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

    def __init__(self, action_size, buffer_size, batch_size, seed, params=Params()):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.params = params
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.nstep_exp_buffer = deque(maxlen=self.params.n_step_bootstrap)
        self.seed = random.seed(seed)

    # Save experiences in replay buffer (w/ N-Step Bootstrap)
    def fill_nstep_buffer(self, experiences, clear_nstep_buffer=False):
        
        # if clear_nstep_buffer:
        #     self.nstep_exp_buffer.clear()

        # self.nstep_exp_buffer.append(experiences)
        # if len(self.nstep_exp_buffer) >= self.params.n_step_bootstrap:
            
        #     num_robots = len(experiences[0])
        #     discounted_future_rewards = [0] * num_robots
        #     for i, exp in enumerate(self.nstep_exp_buffer):
        #         discounted_future_rewards += ((self.params.gamma**i) * np.array(exp[2]))
            
        #     initial_exp = self.nstep_exp_buffer.popleft()
        #     for i in range(num_robots):
        #         self.add(initial_exp[0][i], initial_exp[1][i], discounted_future_rewards[i], experiences[3][i], experiences[4][i])
            
        self.nstep_exp_buffer.append(experiences)

        # Abort if ROLLOUT steps haven't been taken in a new episode
        if len(self.nstep_exp_buffer) < self.params.n_step_bootstrap:
            return

        # Unpacks and stores the SARS' tuple for each actor in the environment
        # thus, each timestep actually adds K_ACTORS memories to the buffer,
        # for the Udacity environment this means 20 memories each timestep.
        
        for actor in zip(*self.nstep_exp_buffer):

            states, actions, rewards, next_states, dones = zip(*actor)
            n_steps = self.params.n_step_bootstrap

            # Calculate n-step discounted reward
            rewards = np.fromiter((self.params.gamma**i * rewards[i] for i in range(n_steps)), float, count=n_steps)
            rewards = rewards.sum()

            # store the current state, current action, cumulative discounted
            # reward from t -> t+n-1, and the next_state at t+n (S't+n)
            states = states[0]
            actions = torch.from_numpy(actions[0]).double()
            rewards = torch.tensor([rewards])
            next_states = next_states[-1]
            self.add(states, actions, rewards, next_states, dones)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
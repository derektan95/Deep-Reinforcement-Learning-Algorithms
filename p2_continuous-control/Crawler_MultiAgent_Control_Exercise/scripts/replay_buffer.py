from collections import deque, namedtuple
from params import Params
import numpy as np
import random
import torch

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
        self.device = self.params.device
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.nstep_exp_buffer = deque(maxlen=self.params.n_step_bootstrap)
        self.seed = random.seed(seed)

    # Save experiences in replay buffer (w/ N-Step Bootstrap)
    def fill_nstep_buffer(self, experiences, clear_nstep_buffer=False):
        
        if clear_nstep_buffer:
            self.nstep_exp_buffer.clear()

        self.nstep_exp_buffer.append(experiences)
        if len(self.nstep_exp_buffer) >= self.params.n_step_bootstrap:
            
            num_robots = len(experiences[0])
            discounted_future_rewards = [0] * num_robots
            for i, exp in enumerate(self.nstep_exp_buffer):
                discounted_future_rewards += ((self.params.gamma**i) * np.array(exp[2]))
            
            initial_exp = self.nstep_exp_buffer.popleft()
            for i in range(num_robots):
                self.add(initial_exp[0][i], initial_exp[1][i], discounted_future_rewards[i], experiences[3][i], experiences[4][i])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
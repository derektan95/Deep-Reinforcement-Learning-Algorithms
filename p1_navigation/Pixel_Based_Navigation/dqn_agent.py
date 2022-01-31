import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)              # replay buffer size
BATCH_SIZE = 64                     # minibatch size
GAMMA = 0.99                        # discount factor
TAU = 1e-3                          # for soft update of target parameters
LR = 5e-4                           # learning rate 
LEARN_EVERY = 4                     # how often to update the LOCAL network
UPDATE_TARGET_WEIGHTS_EVERY = 500   # how often to update the TARGET network


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Initialize time step
        self.learn_t_step = 0       # for updating every LEARN_EVERY steps
        self.soft_update_t_step = 0 # for updating every UPDATE_TARGET_WEIGHTS_EVERY steps


        ## Print networks
        print("network_local", self.qnetwork_local)
        print("network_target", self.qnetwork_target)
        print("optimizer", self.optimizer)
        
        # Print Hyper-parameters
        print("BUFFER_SIZE: ", BUFFER_SIZE)
        print("BATCH_SIZE: ", BATCH_SIZE)
        print("GAMMA: ", GAMMA)
        print("TAU: ", TAU)
        print("LR: ", LR)
        print("LEARN_EVERY: ", LEARN_EVERY)
        print("UPDATE_TARGET_WEIGHTS_EVERY: ", UPDATE_TARGET_WEIGHTS_EVERY)
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every LEARN_EVERY time steps.
        self.learn_t_step = (self.learn_t_step + 1) % LEARN_EVERY
        self.soft_update_t_step = (self.soft_update_t_step + 1) % UPDATE_TARGET_WEIGHTS_EVERY

        if self.learn_t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.qnetwork_local.eval()                           # .eval() == (self.training=false)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)       # INFERENCE: NO NEED TO UPDATE WEIGHTS / BIASES VIA BACKPROP
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"

        # # For greyscale, unsqueeze the 1 dimension that is lost in the process
        # next_states = torch.unsqueeze(next_states, 1)
        # states = torch.unsqueeze(states, 1)
        # print("next_states: ", next_states.shape)
        
        # Target actions from stable Fixed Target-Q Neural Network
        # Detach since no need to update weights & biases param in Target Network - They are cloned from qnetwork_local
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)   
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))   # If done, ignore next action
        
        # Current actions from Q-Approximator Neural Network
        Q_expecteds_arr = self.qnetwork_local(states)
        Q_expecteds = Q_expecteds_arr[torch.arange(Q_expecteds_arr.shape[0]).long(), actions.squeeze().long()].unsqueeze(1)
        
        # Compute & minimize the loss
        loss = F.mse_loss(Q_expecteds, Q_targets)   # Mean-Squared Error loss across mini-batch of experiences relative to targets array
        self.optimizer.zero_grad()                  # Zero out all of the gradients for the variables which the optimizer will update
        loss.backward()                             # Compute the gradient of the loss wrt each parameter of the model.
        self.optimizer.step()                       # Actually update the parameters of the model using the gradients computed by the backwards pass.

        # ------------------- update target network ------------------- #
        if self.soft_update_t_step == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)           

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
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
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

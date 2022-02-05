class Params():
    """Actor (Policy) Model."""

    def __init__(self):

        self.buffer_size = int(3e5)           # replay buffer size
        self.batch_size = 128                 # minibatch size
        self.gamma = 0.99                     # discount factor
        self.tau = 1e-3                       # for soft update of target parameters
        self.lr_actor = 5e-4                  # learning rate of the actor 
        self.lr_critic = 1e-3                 # learning rate of the critic
        self.weight_decay = 1e-4              # L2 weight decay          (ORIGINAL: 0)
        self.hard_update=True                 # Hard update OR Soft Update?
        self.learn_every = 1                  # how often for local networks to learn
        self.soft_weights_update_every = 10   # how often to copy weights over to target networks (Gradually)
        self.hard_weights_update_every = 350  # how often to copy weights over to target networks (Instant)
        self.n_step_bootstrap = 5             # N-Step bootstrapping for Temporal Difference Update Calculations

        # D4PG STUFF
        # Lower and upper bounds of critic value output distribution, these will vary with environment
        # V_min and V_max should be chosen based on the range of normalised reward values in the chosen env
        # Assume Normalized Reward = Rewards / 100
        self.vmin = 0
        self.vmax = 0.3
        self.num_atoms = 100




######## FOR REFERENCE ########
# BUFFER_SIZE = int(1e6)  # replay buffer size
# BATCH_SIZE = 128        # minibatch size
# GAMMA = 0.99            # discount factor
# TAU = 1e-3              # for soft update of target parameters
# LR_ACTOR = 5e-4         # learning rate of the actor 
# LR_CRITIC = 1e-3        # learning rate of the critic
# WEIGHT_DECAY = 1e-2     # L2 weight decay          (ORIGINAL: 0)
# HARD_UPDATE=False       # Hard update OR Soft Update?
# LEARN_EVERY = 1                  # how often for local networks to learn
# SOFT_WEIGHTS_UPDATE_EVERY = 10   # how often to copy weights over to target networks (Gradually)
# HARD_WEIGHTS_UPDATE_EVERY = 350  # how often to copy weights over to target networks (Instant)
# N_STEP_BOOTSTRAP = 5             # N-Step bootstrapping for Temporal Difference Update Calculations

# # D4PG STUFF
# # Lower and upper bounds of critic value output distribution, these will vary with environment
# # V_min and V_max should be chosen based on the range of normalised reward values in the chosen env
# # Assume Normalized Reward = Rewards / 100
# VMIN = 0
# VMAX = 0.3
# NUM_ATOMS = 100

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ATOMS = torch.linspace(self.params.vmin, self.params.vmax, self.params.num_atoms).to(DEVICE)
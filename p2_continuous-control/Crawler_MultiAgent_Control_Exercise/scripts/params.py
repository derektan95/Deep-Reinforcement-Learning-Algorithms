import torch

class Params():
    """Actor (Policy) Model."""

    def __init__(self):

        # Simulation Based Params
        self.random_seed = 0                     # Random seed to start sim on
        self.n_episodes = 200                    # Number of episodes to run sim for
        self.max_t = 1000                        # Max sim step before episode terminates
        self.print_every = 100                   # Prints every x episodes
        self.save_every = 100                    # Saves weights every x episodes
        self.log_weights_every = 1              # How often to log weights in Tensorboard
        self.terminate_on_target_score = False   # Terminates simulation upon reaching target score
        self.target_score = 1800                 # Target score to achieve before sim termination 
        self.prefill_memory_qty = 5000           # Experience (SARS) qty to prefill replay buffer before training
        self.verbose = False                     # Whether to print debug messages
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # General Hyper-params
        self.buffer_size = int(3e5)              # replay buffer size
        self.batch_size = 128                    # minibatch size
        self.hidden_sizes=(256, 128)             # Hidden layer sizes
        self.gamma = 0.99                        # discount factor
        self.tau = 1e-3                          # for soft update of target parameters
        self.lr_actor = 5e-4                     # learning rate of the actor 
        self.lr_critic = 1e-3                    # learning rate of the critic
        self.weight_decay = 1e-4                 # L2 weight decay          (ORIGINAL: 0)
        self.hard_update=True                    # Hard update OR Soft Update?
        self.learn_every = 1                     # how often for local networks to learn
        self.soft_weights_update_every = 10      # how often to copy weights over to target networks (Gradually)
        self.hard_weights_update_every = 350     # how often to copy weights over to target networks (Instant)
        self.n_step_bootstrap = 5                # N-Step bootstrapping for Temporal Difference Update Calculations

        # Misc
        self.checkpoint_actor_weights_dir = 'weights/checkpoint_actor'
        self.checkpoint_critic_weights_dir = 'weights/checkpoint_critic'

        # D4PG STUFF
        # Lower and upper bounds of critic value output distribution, these will vary with environment
        # V_min and V_max should be chosen based on the range of normalised reward values in the chosen env
        # Assume Normalized Reward = Rewards / 100
        self.vmin = -0.01                        # Min value which atom's action-value function begins
        self.vmax = 20                           # Max value which atom's action-value function begins 
        self.num_atoms = 100                     # Multiple atoms can represent the distribution of V 


    def print_init_messages(self):
        
        # Print Hyper-parameters
        print("\n=============== HYPERPARAMS ===============")
        print("DEVICE: ", self.device)
        print("RANDOM SEED: ", self.random_seed)
        print("BUFFER_SIZE: ", self.buffer_size)
        print("BATCH_SIZE: ", self.batch_size)
        print("HIDDEN_SIZES: ", self.hidden_sizes)
        print("GAMMA: ", self.gamma)
        print("TAU: ", self.tau)
        print("LR_ACTOR: ", self.lr_actor)
        print("LR_CRITIC: ", self.lr_critic)
        print("WEIGHT_DECAY: ", self.weight_decay)
        print("HARD_UPDATE: ", self.hard_update)
        print("LEARN_EVERY: ", self.learn_every)
        if (self.hard_update):
            print("HARD_WEIGHTS_UPDATE_EVERY: ", self.hard_weights_update_every)
        else:
            print("SOFT_WEIGHTS_UPDATE_EVERY: ", self.soft_weights_update_every)
        print("N_STEP_BOOTSTRAP: ", self.n_step_bootstrap)
        print("VMIN: ", self.vmin)
        print("VMAX: ", self.vmax)
        print("NUM_ATOMS: ", self.num_atoms)
        print("===========================================\n")


    def get_hparam_dict(self):
        """ For tensorboard tracking of impt hyper-params. """
        hparam_dict = {"batch_size": self.batch_size, "actor_lr": self.lr_actor, "critic_lr": self.lr_critic,
                       "hard_update_every": self.hard_weights_update_every, "vmin": self.vmin, "vmax": self.vmax, 
                       "num_atoms": self.num_atoms}

        return hparam_dict

    # def get_hparam_comment(self):
    #     """ 
    #     Generates runfile tensorboard folder name.
    #     NOTE: For some reason, tb doesn't accept folder names that are too lengthy. Must limit number of hyperparams.
    #     """

    #     comment = f'bs={self.batch_size} a_lr={self.lr_actor} c_lr={self.lr_critic} update_every={self.hard_weights_update_every} vmax={self.vmax} vmin={self.vmin}'
    #     return comment
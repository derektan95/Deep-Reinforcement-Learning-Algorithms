import torch

class Params():
    """Actor (Policy) Model."""

    def __init__(self):

        # Simulation Based Params
        self.random_seed = 1                     # Random seed to start sim on
        self.n_episodes = 500                    # Number of episodes to run sim for
        self.max_t = 600                         # Max sim step before episode terminates
        self.print_every = 20                    # Prints every x episodes
        self.save_every = 20                     # Saves weights every x episodes
        self.log_weights_every = 20              # How often to log weights in Tensorboard
        self.plot_stats = True                   # Plot graphs from loggers?
        self.terminate_on_target_score = True    # Terminates simulation upon reaching target score
        self.target_score = 110.0                # Target score to achieve before sim termination 
        self.prefill_memory_qty = 0              # Experience (SARS) qty to prefill replay buffer before training
        self.verbose = True                      # Whether to print debug messages
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # General Hyper-params
        self.batch_size = 32                     # minibatch size
        self.hidden_sizes_actor=(256, 128)       # Hidden layer sizes (Actor Net)
        self.hidden_sizes_critic=(256, 128)      # Hidden layer sizes (Critic Net)
        self.lr = 8e-5                           # learning rate of the actor 
        self.gamma = 0.995                       # discount factor
        self.eps = 0.1                           # How much to clip advantage function
        self.eps_decay = 1                       # How fast to tighten the clipping function
        self.beta = 0.001                        # Entropy to add to the loss fn for exploration (High entropy = more equiprobable)
        self.beta_decay = 0.995                  # How fast to reduce added entropy (Exploitation Rate)
        self.weight_decay = 0                    # L2 weight decay          (ORIGINAL: 0)
        self.n_step_bootstrap = 1                # N-Step bootstrapping for Temporal Difference Update Calculations
        self.gradient_clip = int(0)              # [int(0) to disable] Whether to clip gradient for optimizer to perform backprop
        self.optimizer_eps = 1e-8                # Optimizer epsilon: Term added to denominator for numerical stability

        # Misc
        self.checkpoint_actor_weights_dir = 'weights/checkpoint_actor'
        self.checkpoint_critic_weights_dir = 'weights/checkpoint_critic'
        self.restart_training = True

        # # Restart training params (if restart training is false)
        # self.eps_to_resume_from = 257
        # self.actor_weights_filename_to_resume = 'checkpoint_actor_ep257.pth'
        # self.critic_weights_filename_to_resume = 'checkpoint_critic_ep257.pth'


    # If wanna print all local vars in class, consider 'pprint(vars(self))'
    def print_init_messages(self):
        
        # Print Hyper-parameters
        print("\n=============== HYPERPARAMS ===============")
        print("DEVICE: ", self.device)
        print("RANDOM SEED: ", self.random_seed)
        print("BATCH_SIZE: ", self.batch_size)
        print("HIDDEN_SIZES (ACTOR): ", self.hidden_sizes_actor)
        print("HIDDEN_SIZES (CRITIC): ", self.hidden_sizes_critic)
        print("LR (Joint): ", self.lr)
        print("GAMMA: ", self.gamma)
        print("BETA: ", self.beta)
        print("BETA_DECAY: ", self.beta_decay)
        print("EPS: ", self.eps)
        print("EPS_DECAY: ", self.eps_decay)
        print("WEIGHT_DECAY: ", self.weight_decay)
        print("===========================================\n")


    def get_hparam_dict(self):
        """ For tensorboard tracking of impt hyper-params. """
        hparam_dict = {"batch_size": self.batch_size, "lr": self.lr,
                       "gamma": self.gamma, "beta": self.beta, "beta_decay": self.beta_decay, "eps": self.eps, 
                       "eps_decay": self.eps_decay}

        return hparam_dict

    # def get_hparam_comment(self):
    #     """ 
    #     Generates runfile tensorboard folder name.
    #     NOTE: For some reason, tb doesn't accept folder names that are too lengthy. Must limit number of hyperparams.
    #     """

    #     comment = f'bs={self.batch_size} a_lr={self.lr_actor} c_lr={self.lr_critic} update_every={self.hard_weights_update_every} vmax={self.vmax} vmin={self.vmin}'
    #     return comment
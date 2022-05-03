import torch

class Params():
    """Actor (Policy) Model."""

    def __init__(self):

        # Simulation Based Params
        self.random_seed = 1                        # Random seed to start sim on
        self.n_episodes = 2000                      # Number of episodes to run sim for
        self.max_t = 1000                           # Max sim step before episode terminates (Max for thie env = 1000)
        self.print_every = 20                       # Prints every x episodes
        self.save_every = 20                        # Saves weights every x episodes
        self.log_weights_every = 100                # How often to log weights in Tensorboard
        self.plot_stats = True                      # Plot graphs from loggers?
        self.terminate_on_target_score = True       # Terminates simulation upon reaching target score
        self.target_score = 3000.0                  # Target score to achieve before sim termination 
        self.verbose = True                         # Whether to print debug messages
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # General Hyper-params
        self.num_steps_collect_data = 1000          # Number of steps while collecting data in 1 episode (x12 robots = Total experience tuples)
        self.batch_size = 1024                      # minibatch size
        self.hidden_sizes_actor=(1024, 1024, 512)   # Hidden layer sizes (Actor Net)
        self.hidden_sizes_critic=(1024, 1024, 512)  # Hidden layer sizes (Critic Net)
        self.gamma = 0.95                           # discount factor
        self.lr = 1e-4                              # learning rate of the network 
        self.lr_decay = 0.995                       # Decay rate of LR
        self.lr_min = 1e-5                          # Min value for LR
        self.eps = 0.1                              # How much to clip advantage function
        self.eps_decay = 0.995                      # How fast to tighten the clipping function
        self.eps_min = 0.0                          # Min eps to decay to 
        self.beta = 0.01                            # Entropy to add to the loss fn for exploration (High entropy = more equiprobable)
        self.beta_decay = 0.9975                    # How fast to reduce added entropy (Exploitation Rate)
        self.beta_min = 0.0                         # Min beta to decay to
        self.std_scale = 1.0                        # Initial value of std scale for action resampling
        self.std_scale_decay = 0.995                # How fast to decay std_scale value
        self.std_scale_min = 0.0                    # Min std_scale to decay to
        self.critic_loss_coeff = 0.5                # Coefficient to multiply critic loss by while computing overall loss
        self.weight_decay = 1e-4                    # L2 weight decay          (ORIGINAL: 0)
        self.gradient_clip = 1.0                    # [int(0) to disable] Whether to clip gradient for optimizer to perform backprop
        self.optimizer_eps = 1e-5                   # Optimizer epsilon: Term added to denominator for numerical stability
        self.use_gae = True                         # Whether to use Generalized Advantage Estimation to compute advantage
        self.gae_tau = 0.99                         # GAE's expotential weight discount factor

        # Misc
        self.checkpoint_actor_weights_dir = 'weights/checkpoint_actor'
        self.checkpoint_critic_weights_dir = 'weights/checkpoint_critic'
        self.restart_training = True

        # Restart training params (if restart training is false)
        self.eps_to_resume_from = 257
        self.actor_weights_filename_to_resume = 'checkpoint_actor_ep257.pth'
        self.critic_weights_filename_to_resume = 'checkpoint_critic_ep257.pth'


    # If wanna print all local vars in class, consider 'pprint(vars(self))'
    def print_init_messages(self, agent_ns=""):
        
        # Print Hyper-parameters
        print(f"\n============ {agent_ns} HYPERPARAMS ============")
        print("DEVICE: ", self.device)
        print("RANDOM SEED: ", self.random_seed)
        print("BATCH_SIZE: ", self.batch_size)
        print("HIDDEN_SIZES (ACTOR): ", self.hidden_sizes_actor)
        print("HIDDEN_SIZES (CRITIC): ", self.hidden_sizes_critic)
        print("GAMMA: ", self.gamma)
        print("LR (Joint): ", self.lr)
        print("LR_DECAY: ", self.lr_decay)
        print("LR_MIN: ", self.lr_min)
        print("BETA: ", self.beta)
        print("BETA_DECAY: ", self.beta_decay)
        print("BETA_MIN: ", self.beta_min)
        print("EPS: ", self.eps)
        print("EPS_DECAY: ", self.eps_decay)
        print("EPS_MIN: ", self.eps_min)
        print("WEIGHT_DECAY: ", self.weight_decay)
        print("USE GAE: ", self.use_gae)
        if self.use_gae:
            print("GAE TAU: ", self.gae_tau)
        if self.gradient_clip != 0:
            print("GRAD_CLIP: ", self.gradient_clip)
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
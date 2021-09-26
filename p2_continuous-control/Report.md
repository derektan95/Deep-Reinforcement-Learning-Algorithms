# Project 2 Report: Deep Deterministic Policy Gradients (DDPG) for Robot Arm Continuous Control

## Summary of Content
- [Deep Q-Learning Description](#learning-algorithm)
- [Hyperparameters Chosen](#hyperparameters-chosen)
- [Results](#results)
- [Ideas for Future Work](#ideas-for-future-work)

## Learning Algorithm 
The key learning algorithm used in this project is Deep Deterministic Policy Gradients (DDPG). We have to first gain a deeper understanding of policy gradients and actor-critic algorithms before we can better appreciate the use of DDPG in solving our robot navigation task. 

### Policy-Based Methods vs Value-Based Methods

With Policy-Based methods, agents directly learn the optimal policy. In other words, policy-based agents simply takes in the state of the environment and decides to take the action that will yield the highest rewards. With Value-Based methods, on the other hand, agents uses its experience with the environment to maintain an estimate of the optimal action-value function. The optimal policy is then obtained from the optimal action-value function estimate. Policy-based methods performs network updates using policy-gradient methods, which involves the update of netowrk weights in a direction that increases the likelihood of the network outputting actions that maximizes rewards.  


### Actor-Critic Algorithms
Actor-Critic algorithms is a class of algorithms that combines both policy-based and value-based networks. Policy-based networks (Actor) tend to have low bias and high variance, while Value-based networks (Critic) tend to have high bias and low variance (Particularly, Temporal Difference Estimate). Therefore, both types of networks are combined in an attempt to achieve the best of both worlds: low bias and low variance. This will allow for more stable training and faster convergence.  


### Deep Deterministic Policy Gradients

### Experience Replay

### DDPG Network Weights Update



<br>



#### Experience Replay


<p align="center">
  <img src="media/dqn_replay_buffer.PNG" width="900" height="300" />
</p>

#### Fixed Q-Targets


<p align="center">
  <img src="media/dqn_policy_update.PNG" width="550" height="300" />
</p>



<br>

### Hyperparameters Chosen
1) **Neural Network Model:** Linear (64) - ReLU - Linear(64) - ReLU - Linear (4)
2) **Episodes:** 1800
3) **Max Duration:** 1000 timesteps  &nbsp;  
4) **Epsilon (Start):** 1.0                 
5) **Epsilon (End):** 0.01                   
6) **Epsilon Decay Rate:** 0.995            
7) **Replay Buffer Size (Max)**: 1e5         
8) **Buffer Batch Size (Sample)**: 64        
9) **Discount Factor (Gamma)**: 0.99        
10) **Target Param Update Rate (Tau)**: 1e-3        
11) **Learning Rate (Optimizer)**: 5e-4       
12) **Update Every (Learning)**: 4


### Results
The results below is obtained from my implementation of Deep Q-Learning for this project. As you can see, training stabilizes early around 800 episodes. Training achieves a **score higher than +13 after 532 episodes (Average between 432-532)** (verify on [notebook](https://github.com/derektan95/deep-reinforcement-learning-udacity-nanodegree/blob/master/p1_navigation/Navigation.ipynb)).

<p align="center">
  <img src="media/score_vs_episodes_dqn.PNG" width="500" height="300" />
</p>


### Ideas for Future Work
There are 2 possible improvements that could be made to the DDPG algorithm. 

#### Generalized Advantage Functions


<p align="center">
  <img src="media/double_dqn.PNG" width="700" height="230" />
</p>

#### Synchronous Replay Buffer


<p align="center">
  <img src="media/prioritized_experience_replay.PNG" width="900" height="300" />
</p>

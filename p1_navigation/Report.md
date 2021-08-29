# Project 1 Report: Deep Q-Learning for Robot Navigation

## Learning Algorithm 
The key learning algorithm used in this project is Deep Q-Learing. We have to first gain a deeper understanding of conventional Reinforcement Learning algorithms before we can better  appreciate the use of Deep Q-Learning in solving our robot navigation task. 

### Conventional Q-Learning

Conventional Reinforcement Learning algorithms often involve 2 key stages at each time step. The agent will firstly select an action using the **epsilon-greedy policy** (Policy Improvement),  followed by **updating its Q-table** as it accumulates experience through the epsidoes it has been through (Policy Evaluation). 

<img src="media/policy_evalulation_and_improvement.PNG" width="900" height="400" />

#### Epsilon Greedy Policies (Policy Improvement)
**Epsilon Greedy policies** balances out the extent to which the agent is exploratory vs exploitative. More specifically, it balances out the agent desires to explore the environment more or to  choose the action greedily that would lead to highest expected cumulative rewards from their current understanding of the world. It is often desirable for the agent to explore the environment more in the early episodes, and to greedily choose actions in the later episodes. An epsilon-greedy policy can be represented by the equation below, assuming that the state space is discrete. 

<img src="media/epsilon_greedy_policy.PNG" width="900" height="200" />
<img src="media/epsilon_greedy_policy_equation.PNG" width="900" height="200" />

#### Q-Learning (Policy Evaluation)
An agent **updates its Q-table** by following the equation below. This equation is specific to SARSA-Max, which is also known as Q-Learning. This is considered an **off-policy** method,  where the greedy policy that is evaluated and improved is different from the epsilon-greedy policy that is used to select action. 

<img src="media/sarsa_max_qlearning_equation.PNG" width="900" height="200" />

The pseudo-code for SARSA-Max (Q-Learning) is shown in the image below: 

<img src="media/sarsa_max_qlearning_pseudo_code.png" width="900" height="300" />

#### Limitations to Conventional RL Approaches
There are **some limitations to conventional approaches** like SARSA-Max. The main reason why SARSA-Max is rarely used in practice is because it is only meant for agents in discrete state spaces. While we could discretize continuous spaces using methods like tile-coding, it is computationally expensive to compute optimal policies in these large discretized state spaces.  There is therefore a need to learn a function approximator to obtain Q-values given the agent's state, through the use of Neural Networks. 


### Features of Deep Q-Learning
There are 2 main features of Deep Q-Learning to mitigate training instability arising from its use of Neural Networks. More specifically, Deep Q-Learning uses 
**Experience Replay** and **Fixed Q-Targets** to achieve training stability. 

#### Experience Replay
If we were to naively follow the conventional Q-Learning approach, the agent would learn from consequetive experiences within the same episodes. This can be detrimental if 
the agent were to experience highly correlated tuples (state, action, reward, next_state) that would lead to instability during training. The use of a replay buffer stores experiences obtained from training episodes. These episodes can be randomly sampled anytime for the agent to learn from. Since these episodes are randomized, the agent will end up learning  from diverse examples, breaking correlations and leading to more stable training. 

<img src="media/dqn_replay_buffer.PNG" width="900" height="300" />


#### Fixed Q-Targets
The agent's objective function is defined to allow our function approximator to be as close to the true optimal policy as possible. We can compute our ideal change in weight parameters to get closer to our goal of having a good q-value approximator, using **Gradient Descent formulation** as shown below. Mathematically, q_pi is not a 1-to-1 replacement for our defined policy update method as they don’t depend on the same parameters. Nevertheless, this allows our function approximator network to learn to grow incrementally towards a better approximation of the optimal policy by 'chasing' the epsilon-greedy policy update.

<img src="media/dqn_policy_update.PNG" width="900" height="300" />

However, since weight parameters are continuously updated to improve our Neural Network derived Q-value, the target itself (epsilon-greedy derived Q-value) continuously changes too. This problem is often described as a car chasing its tail, which leads to high training instability. To fix this, we can introduce a separate neural network that has frozen weights `W(-)`. From time to time, W(-) is copied over from the original neural network W and will then be kept frozen for the original network to chase. This approach geenrates a table target for our function approximator to follow.

<img src="media/dqn_fixed_q_target.PNG" width="900" height="300" />

### Results
The results below is obtained from my implementation of Deep Q-Learning for this project. As you can see, training stabilizes early around 800 episodes. 

<img src="media/score_vs_episodes_dqn.PNG" width="900" height="300" />



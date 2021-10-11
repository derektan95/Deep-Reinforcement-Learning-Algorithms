# Deep Reinforcement Learning Nanodegree (Udacity)

**Disclaimer:** Udacity provided some starter code, but the implementation for these concepts are done by myself. Please contact derektan95@hotmail.com for any questions. <br><br>
**Note:** Please refer to the instructions on how to download the dependencies for these projects [here](https://github.com/derektan95/deep-reinforcement-learning-udacity-nanodegree/blob/master/INSTRUCTIONS.md).

### Certificate of Completion<br/>
https://confirm.udacity.com/XLGDCKNX

## Project Reports
- [Deep Q-Learning](https://github.com/derektan95/deep-reinforcement-learning-udacity-nanodegree/tree/master/p1_navigation)
- [Deep Deterministic Policy Gradient (DDPG)](https://github.com/derektan95/deep-reinforcement-learning-udacity-nanodegree/tree/master/p2_continuous-control)
- [Multi-Agent DDPG](https://github.com/derektan95/deep-reinforcement-learning-udacity-nanodegree/tree/master/p2_continuous-control)

### Summary<br/>
Deep reinforcement learning (deep RL) is a subfield of machine learning that combines reinforcement learning (RL) and deep learning. RL considers the problem of a computational agent learning to make decisions by trial and error. Deep RL incorporates deep learning into the solution, allowing agents to make decisions from unstructured input data without manual engineering of the state space. Deep RL algorithms are able to take in very large inputs (e.g. every pixel rendered to the screen in a video game) and decide what actions to perform to optimize an objective (eg. maximizing the game score). Deep reinforcement learning has been used for a diverse set of applications including but not limited to robotics, video games, natural language processing, computer vision, education, transportation, finance and healthcare.[[1]](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)


<!--## Introduction to Reinforcement Learning <br/>
**Gazebo** is a useful simulation tool that can be used with ROS to render robots in a simulated environment. It comes with a model and world editor, along with presets models, that can allow for quick prototyping of a physical environment.

The main principles taught in this segment are: 
1) Using model editor tool to render a robot with specified links & joints
2) Using World editor tool to render an environment (e.g. a house)
3) Running plugins on launch of Gazebo platform -->


## Deep Q-Learning for Robot Navigation <br/>
Detailed information about the training algorithm and project environment can be found [here](https://github.com/derektan95/deep-reinforcement-learning-udacity-nanodegree/tree/master/p1_navigation).

<!-- **Robot Operating System** is a middleware designed for communication between different robotic parts, as well as common useful packages that can be used for robotic applications. In this project, different communication models were employed for different nodes of the robot to allow the robot to drive towards a white ball whenever the robot observes it. From a high level, the 2D camera node continuously checks whether the white ball in sight, and the angle of the ball relative to the robot's heading. If white ball is in sight, a service is called to the drive node to drive towards the ball with specified linear and rotational velocity. The drive node receives this service call and publishes motion information robot's wheel actuation node for movement. 

The main principles taught in this segment are: 
1) Packages & Catkin Workspaces
2) Writing ROS nodes & communication models (Publisher-Subscriber, Service-Client) -->

<p align="center">
  <img src="p1_navigation/media/p1_dqn_navigation_trained_agent_raw_Trimmed.gif" width="350" height="250" />
  <img src="p1_navigation/media/score_vs_episodes_dqn.PNG" width="400" height="250" />
</p>

## Deep Deterministic Policy Gradient for Robot Arm Continuous Control <br/>
Detailed information about the training algorithm and project environment can be found [here](https://github.com/derektan95/deep-reinforcement-learning-udacity-nanodegree/tree/master/p2_continuous-control).

<!-- 2 common localization principles are the **Extended Kalman Filter (EKF)** and **Monte Carlo Localization (Particle Filter)**. Given a map of the surrounding, motor controls and sensor readings, the robot can use either of these principles to estimate its state position. In this project, I made use of the **Adaptive Monte Carlo Package** from ROS (http://wiki.ros.org/amcl). The robot starts off with a known map, with particles of equal probability weights generated randomly around the robot (shown as **green arrows**). As the robot moves, the particles likewise move. Each particle will then be assigned a probability denoting the likelihood of it being in its position and orientation, by comparing laser distance readings and the distance between it's own position to landmarks on the map. The higher the probability, the more likely a particle will survive in the resampling stage. After multiple timesteps of movement, we can observe that the **green arrows** converges accurately on the true location on the robot, indicating precise localization. 

The main principles taught in this segment are: 
1) Extended Kalman Filter
2) Adaptive Monte Carlos Localization (Particle Filter) -->

<p align="center">
  <img src="p2_continuous-control/media/p2_ddpg_continuous_control_trained_agent_raw_Trimmed.gif" width="350" height="250" />
  <img src="p2_continuous-control/media/ddpg_reward_episode_graph.png" width="400" height="250" />
</p>

## Multi-Agent Deep Deterministic Policy Gradient for Cooperative Tennis <br/>
Detailed information about the training algorithm and project environment can be found [here](https://github.com/derektan95/deep-reinforcement-learning-udacity-nanodegree/tree/master/p3_collab-compet).

<!-- A common mapping algorithm is the **Occupancy Grid Mapping**. Using sensor measurements and the Binary Bayes Filter, it computes the likelihood of an obstacle (i.e. map) given a particular grid on the map. Mapping requires the knowledge of the robot's start position, motor controls and sensor readings.

**Simulataneous Localization and Mapping (SLAM)** combines principles from both localization and mapping. Using sensor readings and motor control, the robot can continuously map the surroundings, and use the map data to localize itself relative to it. The **Online SLAM approach** gives the map and robot's pose at a given point of time, while the **Full SLAM approach** gives the map and all past robot poses. The main techniques taught in this class is the **Grid-Based FastSLAM** and **GraphSLAM**, which are Online Slam and Full Slam approaches respectively. In this project, the **Real Time Appearance Based Mapping** is used as part of the Online SLAM approach, where a depth camera is used. It provides **3D localization and mapping**, with the ability to perform **loop closure** (i.e. identify previously visited locations to allow for smoother map generation). 

This is available as a ROS package (http://wiki.ros.org/rtabmap_ros). Please build the rtab package from source by following the instructions in the RTAB-github link (https://github.com/introlab/rtabmap_ros).

The main principles taught in this segment are: 
1) Occupancy Grid Mapping (Binary Bayes Filter)
2) Grid-Based FastSLAM
3) GraphSLAM
4) RTAB-map SLAM (Variant of GraphSLAM) -->

<p align="center">
  <img src="p3_collab-compet/media/p3_maddpg_tennis_trained_agent_trimmed.gif" width="350" height="250" />
  <img src="p3_collab-compet/media/training_score_maddpg_self_play.png" width="400" height="250" />
</p>

https://www.geeksforgeeks.org/how-to-make-a-reward-function-in-reinforcement-learning/

how to make a reward function in reinforcement learning

the reward model in rl drives agent's learning process by providing feedback on the actions it takes, guiding it towards achieving the desired outcomes.

crafting a proper reward function is essential to ensure that the agent learns the correct behavior.

## understanding the role

in reinforcement learning, an agent's goal is to maximize the cumulative reward over time, known as the return.  

the reward function provides immediate feedback by assigning a numerical value to each action taken by the agent.

the agent learns to perform actions that result in higher rewards by exploring various action-state pairs and updating its policy.

the design of teh reward function significantly influences the agent's behavior. a well designed reward function will lead the agent to solve the problem effectively, *while a poorly designed one may result in undesirable or suboptimal behavior*.

## steps to designing a reward function

### define the goal of the agent

before defining the reward function, it's essential to clearly understand the goal of the agent.

**ask yourself**

* what **specific outcome** should the agent aim to achieve
* what are the **key ebhaviors** you want the agent to learn

*in autonomous driving predict and planning, this means more like human behaviors and avoid crashing, more comfortable, and so on*

for example, if you are working on a game environment, the goal could be winning the game, avoiding obstacles, or reaching a destination in the least amount of time.

## identify positive and negative rewards

once you've identified the goal, determine the actions or states that should result in positive rewards(rewarding desirable behavior) and negative rewards(penelizing undesirable behavior).

* positive rewards should be given when the agent takes actions that bring it closer to the goal. for instance, in a maze solving problem, the agent could receive positive rewards for moving closer to the exit.

* negative rewards or penalties can discourage the agent from taking incorrect actions. for example, in a self-driving car simulation, negative rewards could be assigned for collisions or driving off-road.

## ensure consistency in rewards

it is essential to ensure that the reward values are consistent and aligned with the objective. if someactions yield disproportionately high or low rewards compared to others, it may cause the agent to focus on those actions, leading to suboptimal learning.

for instance, in a grid-world environment, if the reward for reaching the goal is 10, but the penalty for hitting a wall is -100, the agent maay over-focus on avoiding walls rather than efficiently reaching the goal.

## balance immediate and long-term rewards

design the reward function to balance immediate and long-term rewards. immediate rewards may prompt the agent to **take quick, beneficial actions**, but long term rewards help the agent **plan for the future**.

immediate rewards -- take quick, beneficial actions
long-term rewards -- help the agent plan for the future

> for example, in a game where collecting points is the goal, providing small, incremental rewards for each point collected and a large reward for completing the game could motivate the agent to prioritize both short term and long term gains.

## avoid reward hacking

reward hacking occurs when the agent finds a way to achieve high rewards without necessarily achieving the intended goal.

to avoid this, ensure the reward function is well-defined and robust.

an example of reward hacking might occur in a robotic vacuum cleaner simulation, where the robot receives rewards for cleaning certain areas. if not properly designed, the agent may repeatedly clean the same spot to maximize its reward, the agent may repeatedly clean the same spot to maximize its reward, even if the entire environment isn't clean.

> **the difference between reward function and reward model is that the function needs to be well-defined manually but we can train a reward model in a deep learning style.  --li ruiqin**

## common approaches for reward function design

### sparse vs. dense rewards

sparse rewards provide feedback only when a significant event occurs. e.g. the agent only receives a reward when it reaches the goal. sparse rewards can make learning challenging but encourage exploration.

dense rewards provide feedback at every step. e.g. the agent receives a small reward for each correct action that moves it closer to the goal. 

dense rewards facilitate *faster learning* but may lead to *prenmature convergence* on suboptimal strategies.

### shaping rewards

reward shaping involves designing incremental rewards to guide the agent more effectively toward the final goal. for instance, in a robotic navigation task, rather than only rewarding the agent when it reaches the destination, you could shape the rewards by providing small positive rewards for every step taken in the right direction.

## implementing a reward funcion in python

an example --

the agent will navigate towards a goal while avoiding obstacles.

"""python
import numpy as np
import gym

env = gym.make("CartPole-v1")

# define a function to convert continuous state values into discrete bins.
def discretize_state(state, bins):
    return tuple(np.digitize(state[i], bins[i]) -1 for i in range(len(state)))

# q table initialization: initialize the q-table with zeros, using discretized state bins and action space size.
#
# q表是一个数据结构，用于存储状态-动作对的价值。在强化学习中，q表是用于表示状态-动作对的价值函数。
# q表初始化是指将q表初始化为零，使用离散的状态值和动作空间大小。
#
# 常见的q表初始化方法有：
# 1. 随机初始化：将q表中的值随机初始化为一个很小的数，或者负数。
# 2. 零初始化：将q表中的值全部初始化为零。
# 3. 均匀初始化：将q表中的值均匀初始化为一个数，或者负数。
# 4. 优先初始化：根据历史数据或者经验来初始化Q表中的值，帮助算法快速收敛。
state_bins = [
    np.linspace(-4.8, 4.8, 10),
    np.linspace(-4,4,10),
    np.linspace(-0.4188,0.418,10),
    np.linspace(-4,4,10),
    ]

action_space_size = env.action_space.n

q_table = np.zeros([10] * len(state_bins) + [action_space_size])

"""


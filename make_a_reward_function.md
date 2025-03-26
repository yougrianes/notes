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

```python
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

```

### deinfe hyperparameters and reward function

* hyperparameters: set learning rate, discount factor, exploration rate(epsilon), and number of episodes.

* reward function: define a function to provide rewards or penalties based on the agent's actions and outcomes.

```python
# alpha (学习率): 控制Q-learning算法更新Q值的速度。较大的alpha值意味着Q值更新更快，较小的alpha值意味着Q值更新更慢。通常，alpha值在0.1到0.9之间。
alpha = 0.1
# gamma (折扣因子): 控制未来奖励的重要性。gamma值越大，未来奖励越重要；gamma值越小，未来奖励越不重要。通常，gamma值在0.9到0.99之间。
gamma = 0.99
# epsilon (探索率): 控制智能体在选择行动时的随机性。epsilon值越大，智能体越倾向于选择随机行动；epsilon值越小，智能体越倾向于选择贪婪行动（即选择当前估计最优行动）。通常，epsilon值在0.1到1.0之间。
epsilon = 1.0
# epsilon_decay (探索率衰减): 控制epsilon值随着训练迭代次数的增加而衰减的速度。epsilon_decay值越大，epsilon值衰减越慢；epsilon_decay值越小，epsilon值衰减越快。通常，epsilon_decay值在0.99到0.999之间。
epsilon_decay = 0.995
# min_epsilon (最小探索率): 控制epsilon值的最小值。当epsilon值小于min_epsilon时，epsilon值将保持不变。通常，min_epsilon值在0.01到0.1之间。
min_epsilon = 0.01
# episodes (训练迭代次数): 控制Q-learning算法训练的迭代次数。通常，episodes值在100到10000之间。
episodes = 100
```

### q-learning algorithm

1. episode loop, for each episode, reset the environment and initialize variables.
2. action selection, choose an action based on exploration or exploitation.
3. environment interaction, take the action, observe the next state, and calculate the reward.
4. q-table update, update the q-table using the q-learning formula.

# Q 学习算法流程

## 初始化
1. 初始化 Q 值表（$Q(s, a)$）的值。
2. 设置学习率（$\alpha$）、折扣因子（$\gamma$）、探索率（$\epsilon$）和最小探索率（$min_{\epsilon}$）。
3. 设置训练的迭代次数（$episodes$）。

## Episode 开始
1. 初始化当前状态（$s$）和动作（$a$）。
2. 设置当前 episode 的奖励（$R$）为 0。

## 选择动作
1. 以概率 $\epsilon$ 选择一个随机动作（$a$）从可能的动作集合中。
2. 以概率 $1 - \epsilon$ 选择当前状态（$s$）下 Q 值最大的动作（$a$）。

## 执行动作
1. 执行选择的动作（$a$）在当前状态（$s$）。
2. 观察下一个状态（$s'$）和获得奖励（$r$）。
3. 更新当前 episode 的奖励（$R$）通过添加奖励（$r$）。

## 更新 Q 值
### 计算时序差异（TD）误差
$TD_{误差} = r + \gamma * \max(Q(s', a')) - Q(s, a)$

### 更新当前状态（$s$）和动作（$a$）的 Q 值
$Q(s, a) \leftarrow Q(s, a) + \alpha * TD_{误差}$

## 更新探索率
1. 减少探索率（$\epsilon$）通过一个因子 $\epsilon_{decay}$。
2. 如果 $\epsilon < min_{\epsilon}$，则设置 $\epsilon$ 为 $min_{\epsilon}$。

## Episode 结束
1. 如果 episode 结束（例如，智能体到达了终止状态），则重置当前状态（$s$）和动作（$a$）。
2. 重复步骤 3 - 7 直到下一个 episode。

## 重复
重复步骤 2 - 7 直到指定的迭代次数（$episodes$）。



----

20250326
there are some bugs in the original tutorial. first, it doesn't assigned the version of gym, and some api had changede already. second, i use gym 0.17.3, and env.step(action) have different nums of output values to unpack, so it behaves a bit weird.

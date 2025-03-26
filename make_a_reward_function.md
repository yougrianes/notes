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


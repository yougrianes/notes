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

**the difference between reward function and reward model is that the function needs to be well-defined manually but we can train a reward model in a deep learning style.  --li ruiqin**


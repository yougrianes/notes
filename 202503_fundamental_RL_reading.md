# RL fundamental concepts

## intro

1. markov decision processes are intented to include just three aspects, sensation, aaction and goal, a learning agent must be able to sense the state of its environment to some extent and must be able to take actions that affect the state. the agent also must have a goal or goals relating to the state of the environment.

2. reinforcement learning is trying to maximize a reward signal instead of trying to find hidden structure. Uncovering structure in an agent's experience can certainly be useful in reinforcement learning, but by itself does not address the reinforcement learning problem of maximizing a reward signal.

3. Q: what is the difference between rl and supervised / unsupervised learning?

4. the agent has to **exploit** what it has already experienced in order to obtain reward but it also has to **explore** in order to make better action selections in the future.

5. another key feature of reinforcement learning is that it explicitly considers the whole problem of a goal-directed agent interacting with an uncertain environment.

6. Q: some researchers have developed theories of planning with general goals, but without considering planning's role in real-time decision making, or the question of hwere the predictive models necerssary for planning would come from.

7. reinforcement learning takes the opposite tack, starting with a complete, interactive, goal-seeking agent.

8. all reinfoecement learning agents have explicit goals, can sense aspects of their environments, and can choose actions to influence their environments. moreover, it is usually assumed from the beginning that the agent has to operate despite significant uncertainty about the environment it faces. when reinforcement learning involves planning, it has to address the interplay between planning and real-time action selection, as well as the questino of how environment models are acquired and improved.

9. when reinforcement learning involves superviesed learning it does so for specific reasons that determine which capabilities are critical and which are not.

10. one of the most exciting aspects of modern reinforcement learning is its substantive and fruitful interactions with other engineering and scientific disciplines.

12. method based on general principles, such as search or learning, were characterized as *weak methods*, whereas those based on specific knowledge were called *strong methods*.

13. we could choose to exploit most of the time with a small chance of exploring. for we could roll a dice if it lands on one, then we'll explore. otherwise, well choose the greedy action. we call this method epsilon-greedy, where epsilon refers to the probability of choosing to explore.

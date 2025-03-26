import numpy as np
import gym

env = gym.make('CartPole-v1')

# define a function to convert continuous state values into discrete bins.
def discretize_state(state, bins):
    # return tuple(np.digitize(state[i], bins[i]) -1 for i in range(len(state)))
    result = []
    for i, _ in enumerate(state):
        rslt_ = np.digitize(state[i], bins[i])
        result.append(rslt_ - 1)
    return tuple(result)

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


# hyperparameters

# alpha (学习率): 控制Q-learning算法更新Q值的速度。较大的alpha值意味着Q值更新更快，较小的alpha值意味着Q值更新更慢。通常，alpha值在0.1到0.9之间。
alpha = 0.1
# gamma (折扣因子): 控制未来奖励的重要性。gamma值越大，未来奖励越重要；gamma值越小，未来奖励越不重要。通常，gamma值在0.9到0.99之间。
gamma = 0.99
# epsilon (探索率): 控制智能体在选择行动时的随机性。epsilon值越大，智能体越倾向于选择随机行动；epsilon值越小，智能体越倾向于选择贪婪行动（即选择当前估计最优行动）。通常，epsilon值在0.1到1.0之间。
epsilon = 1.0
# epsilon_decay (探索率衰减): 控制epsilon值随着训练迭代次数的增加而衰减的速度。epsilon_decay值越大，epsilon值衰减越慢；epsilon_decay值越小，epsilon值衰减越快。通常，epsilon_decay值在0.99到0.999之间。
epsilon_decay = 0.999
# min_epsilon (最小探索率): 控制epsilon值的最小值。当epsilon值小于min_epsilon时，epsilon值将保持不变。通常，min_epsilon值在0.01到0.1之间。
min_epsilon = 0.01
# episodes (训练迭代次数): 控制Q-learning算法训练的迭代次数。通常，episodes值在100到10000之间。
episodes = 100

def reward_function(state, action, next_state, done):
    if done:
        return -100
    else:
        return 1

# ————————————————————————
# Q-learning algorithm
# ————————————————————————
for episode in range(episodes):
    state = discretize_state(env.reset(), state_bins)
    done = False
    total_reward = 0

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, _, done, truncated = env.step(action)
        next_state = discretize_state(next_state, state_bins)
        reward = reward_function(state, action, next_state, done or truncated)

        q_table[state + (action,)] = q_table[state + (action,)] + alpha * \
            (reward + gamma *
             np.max(q_table[next_state]) - q_table[state + (action,)])

        state = next_state
        total_reward += reward

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

env.close()

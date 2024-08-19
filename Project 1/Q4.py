import numpy as np
from model import Model, Actions
from world_config import cliff_world
import matplotlib.pyplot as plt
from plot_vp import plot_vp

def choose_action(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(list(Actions))
    else:
        return Actions(np.argmax(Q[state]))

def sarsa(model: Model, alpha: float, gamma: float, epsilon: float, episodes: int, max_steps: int, min_epsilon: float):
    Q = np.zeros((model.num_states, len(Actions)))
    rewards_per_episode = []  # 用于存储每轮的累计奖励
    epsilon_decay = (epsilon - min_epsilon) / 100
    for episode in range(episodes):
        state = model.start_state
        action = choose_action(state, Q, epsilon)
        total_reward = 0
        for step in range(max_steps):
            next_state = model.cell2state(model._result_action(model.state2cell(state), action))
            reward = model.reward(state, action)
            next_action = choose_action(next_state, Q, epsilon)
            if episode < 150:
                Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            state, action = next_state, next_action
            if state == model.goal_state:
                break
            total_reward += reward
        rewards_per_episode.append(total_reward)  # 保存每轮的累计奖励
        epsilon = max(min_epsilon, epsilon - epsilon_decay)  # 衰减epsilon
        print("SARSA   ", episode, "   Total reward: ", total_reward)
    policy = np.argmax(Q, axis=1)
    return Q, policy, rewards_per_episode

def q_learning(model: Model, alpha: float, gamma: float, epsilon: float, episodes: int, max_steps: int, min_epsilon: float):
    Q = np.zeros((model.num_states, len(Actions)))
    rewards_per_episode = []  # 用于存储每轮的累计奖励
    epsilon_decay = (epsilon - min_epsilon) / 100
    for episode in range(episodes):
        state = model.start_state
        total_reward = 0
        for step in range(max_steps):
            action = choose_action(state, Q, epsilon)
            next_state = model.cell2state(model._result_action(model.state2cell(state), action))
            reward = model.reward(state, action)
            next_action = np.argmax(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            if state == model.goal_state:
                break
            total_reward += reward
        rewards_per_episode.append(total_reward)  # 保存每轮的累计奖励
        epsilon = max(min_epsilon, epsilon - epsilon_decay)  # 衰减epsilon
        print("Q-Learning   ", episode, "   Total reward: ", total_reward)
    policy = np.argmax(Q, axis=1)
    return Q, policy, rewards_per_episode

if __name__ == "__main__":
    model = Model(cliff_world)

    # SARSA参数
    alpha_sarsa = 0.308  # 较高的学习率
    gamma_sarsa = 0.9  # 折扣因子
    epsilon_sarsa = 0.12  # 初始探索参数
    min_epsilon_sarsa = 0.001  # 最小探索参数
    episodes_sarsa = 250  # 剧集数量
    max_steps_sarsa = 1000  # 每个剧集的最大迭代次数

    # Q-learning参数
    alpha_q_learning = 0.308  # 较低的学习率
    gamma_q_learning = 0.9  # 折扣因子
    epsilon_q_learning = 0.3  # 初始探索参数
    min_epsilon_q_learning = 0.001  # 最小探索参数
    episodes_q_learning = 250  # 剧集数量
    max_steps_q_learning = 1000  # 每个剧集的最大迭代次数

    # 运行SARSA算法
    Q_sarsa, policy_sarsa, rewards_sarsa = sarsa(model, alpha_sarsa, gamma_sarsa, epsilon_sarsa, episodes_sarsa, max_steps_sarsa, min_epsilon_sarsa)
    V_sarsa = np.max(Q_sarsa, axis=1)
    plot_vp(model, V_sarsa, policy_sarsa)
    plt.title("SARSA on Cliff World")
    plt.show()

    # 运行Q学习算法
    Q_q_learning, policy_q_learning, rewards_q_learning = q_learning(model, alpha_q_learning, gamma_q_learning, epsilon_q_learning, episodes_q_learning, max_steps_q_learning, min_epsilon_q_learning)
    V_q_learning = np.max(Q_q_learning, axis=1)
    plot_vp(model, V_q_learning, policy_q_learning)
    plt.ylim(-50, 0)  # 设置y轴的范围
    plt.title("Q-Learning on Cliff World")
    plt.show()

    # 绘制学习曲线
    plt.plot(range(1, len(rewards_q_learning) + 1), rewards_q_learning, label='Q-Learning')
    plt.plot(range(1, len(rewards_sarsa) + 1), rewards_sarsa, label='SARSA')

    plt.xlabel('Episodes')
    plt.ylabel('Total Reward per Episode')
    plt.ylim(-50, 0)  # 设置y轴的范围
    plt.title('Learning Curves for SARSA and Q-Learning on Cliff World')
    plt.legend()
    plt.show()
import numpy as np
from model import Model, Actions
from world_config import *
import matplotlib.pyplot as plt
from plot_vp import plot_vp

def choose_action(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(list(Actions))
    else:
        return Actions(np.argmax(Q[state]))
    
def expected_sarsa(model: Model, alpha: float, gamma: float, epsilon: float, episodes: int, max_steps: int):
    Q = np.zeros((model.num_states, len(Actions)))
    rewards_per_episode = []  # 用于存储每轮的累计奖励
    for episode in range(episodes):
        state = model.start_state
        total_reward = 0
        for step in range(max_steps):
            action = choose_action(state, Q, epsilon)
            next_state = model.cell2state(model._result_action(model.state2cell(state), action))
            reward = model.reward(state, action)
            next_action_probs = np.ones(len(Actions)) * epsilon / len(Actions)
            next_action_probs[np.argmax(Q[next_state])] += 1 - epsilon
            expected_q = np.dot(next_action_probs, Q[next_state])
            Q[state, action] += alpha * (reward + gamma * expected_q - Q[state, action])
            state = next_state
            total_reward += reward
            if state == model.goal_state:
                total_reward += model.reward(state, action)
                break
        rewards_per_episode.append(total_reward)  # 保存每轮的累计奖励
    policy = np.argmax(Q, axis=1)
    return Q, policy, rewards_per_episode

def sarsa(model: Model, alpha: float, gamma: float, epsilon: float, episodes: int, max_steps: int):
    Q = np.zeros((model.num_states, len(Actions)))
    rewards_per_episode = []  # 新增：用于存储每轮的累计奖励
    for episode in range(episodes):
        state = model.start_state
        action = choose_action(state, Q, epsilon)
        total_reward = 0
        for step in range(max_steps):
            next_state = model.cell2state(model._result_action(model.state2cell(state), action))
            reward = model.reward(state, action)
            next_action = choose_action(next_state, Q, epsilon)
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            state, action = next_state, next_action
            total_reward += reward
            print(f"Step {step + 1}: State = {state}, Action = {action}, Reward = {reward}")
            if state == model.goal_state:
                print(f"Goal reached in {step + 1} steps")
                total_reward += model.reward(state, action)
                break
        rewards_per_episode.append(total_reward)  # 新增：保存每轮的累计奖励
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    policy = np.argmax(Q, axis=1)
    return Q, policy,rewards_per_episode


import numpy as np

if __name__ == "__main__":
    model = Model(small_world)
    alpha = 0.308  # 学习率
    gamma = 0.9  # 折扣因子
    epsilon = 0.1  # 探索参数
    episodes = 1000  # 剧集数量
    max_steps = 1000  # 每个剧集的最大迭代次数

    # 运行SARSA算法
    Q_sarsa, policy_sarsa, rewards_sarsa = sarsa(model, alpha, gamma, epsilon, episodes, max_steps)
    V_sarsa = np.max(Q_sarsa, axis=1)
    plot_vp(model, V_sarsa, policy_sarsa)
    plt.title("SARSA on Small World")
    plt.show()

    # 运行Expected SARSA算法
    Q_expected_sarsa, policy_expected_sarsa, rewards_expected_sarsa = expected_sarsa(model, alpha, gamma, epsilon, episodes, max_steps)
    V_expected_sarsa = np.max(Q_expected_sarsa, axis=1)
    plot_vp(model, V_expected_sarsa, policy_expected_sarsa)
    plt.title("Expected SARSA on Small World")
    plt.show()

    # 绘制学习曲线
    plt.plot(range(1, len(rewards_expected_sarsa) + 1), rewards_expected_sarsa, label='Expected SARSA')
    plt.plot(range(1, len(rewards_sarsa) + 1), rewards_sarsa, label='SARSA')

    plt.xlabel('Episodes')
    plt.ylabel('Total Reward per Episode')
    plt.title('Learning Curves for SARSA and Expected SARSA on Small World')
    plt.legend()
    plt.show()


    def smooth_rewards(rewards, window_size=10):
        smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        return smoothed_rewards


    if __name__ == "__main__":
        model = Model(small_world)
        alpha = 0.308  # 学习率
        gamma = 0.9  # 折扣因子
        epsilon = 0.1  # 探索参数
        episodes = 1000  # 剧集数量
        max_steps = 1000  # 每个剧集的最大迭代次数

        # 运行SARSA算法
        Q_sarsa, policy_sarsa, rewards_sarsa = sarsa(model, alpha, gamma, epsilon, episodes, max_steps)
        V_sarsa = np.max(Q_sarsa, axis=1)
        plot_vp(model, V_sarsa, policy_sarsa)
        plt.title("SARSA on Small World")
        plt.show()

        # 运行Expected SARSA算法
        Q_expected_sarsa, policy_expected_sarsa, rewards_expected_sarsa = expected_sarsa(model, alpha, gamma, epsilon,
                                                                                         episodes, max_steps)
        V_expected_sarsa = np.max(Q_expected_sarsa, axis=1)
        plot_vp(model, V_expected_sarsa, policy_expected_sarsa)
        plt.title("Expected SARSA on Small World")
        plt.show()

        # 绘制学习曲线
        smoothed_rewards_sarsa = smooth_rewards(rewards_sarsa)
        smoothed_rewards_expected_sarsa = smooth_rewards(rewards_expected_sarsa)

        plt.plot(range(1, len(smoothed_rewards_sarsa) + 1), smoothed_rewards_sarsa, label='SARSA')
        plt.plot(range(1, len(smoothed_rewards_expected_sarsa) + 1), smoothed_rewards_expected_sarsa,
                 label='Expected SARSA')

        plt.xlabel('Episodes')
        plt.ylabel('Total Reward per Episode')
        plt.title('Learning Curves for SARSA and Expected SARSA on Small World')
        plt.legend()
        plt.show()

        # 计算并打印方差
        variance_sarsa = np.var(rewards_sarsa)
        variance_expected_sarsa = np.var(rewards_expected_sarsa)
        print(f"SARSA Reward Variance: {variance_sarsa}")
        print(f"Expected SARSA Reward Variance: {variance_expected_sarsa}")
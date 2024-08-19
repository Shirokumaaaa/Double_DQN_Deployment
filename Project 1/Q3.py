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

def q_learning(model: Model, alpha: float, gamma: float, epsilon: float, episodes: int, max_steps: int):
    Q = np.zeros((model.num_states, len(Actions)))
    rewards_per_episode = []  # 新增：用于存储每轮的累计奖励
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
            total_reward += reward
            print(f"Step {step + 1}: State = {state}, Action = {action}, Reward = {reward}")
            if state == model.goal_state:
                print(f"Goal reached in {step + 1} steps")
                total_reward += model.reward(state, action)
                break
        rewards_per_episode.append(total_reward)  # 新增：保存每轮的累计奖励
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    policy = np.argmax(Q, axis=1)
    return Q, policy, rewards_per_episode

if __name__ == "__main__":
    model = Model(small_world)
    alpha = 0.308  # 学习率
    gamma = 0.9  # 折扣因子
    epsilon = 0.1  # 探索参数
    episodes = 1000  # 剧集数量
    max_steps = 1000  # 每个剧集的最大迭代次数
    parameters = [
        {'alpha': 0.1, 'gamma': 0.9, 'epsilon': 0.1},
        {'alpha': 0.2, 'gamma': 0.9, 'epsilon': 0.2},
        {'alpha': 0.308, 'gamma': 0.9, 'epsilon': 0.1}
    ]
    Q, policy, reward = q_learning(model, alpha, gamma, epsilon, episodes, max_steps)
    V = np.max(Q, axis=1)
    plot_vp(model, V, policy)
    plt.title("Q-Learning on Small World")
    plt.show()
    # 对于每组参数，运行Q学习并收集数据
    results = []
    for param_set in parameters:
        _, _, rewards = q_learning(model, **param_set, episodes=1000, max_steps=1000)
        results.append(rewards)
    # 绘制每组参数的学习曲线
    for i, rewards in enumerate(results):
        plt.plot(range(1, len(rewards) + 1), rewards, label=f'Set {i + 1}')

    plt.xlabel('Episodes')
    plt.ylabel('Total Reward per Episode')
    plt.title('Learning Curves for Different Parameter Sets')
    plt.legend()
    plt.show()

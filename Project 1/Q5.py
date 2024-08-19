import numpy as np
import matplotlib.pyplot as plt
from model import Model, Actions, Cell
from world_config import WorldConfig
from plot_vp import plot_vp

# 定义small_world变体环境

import torch
import torch.nn as nn
import torch.optim as optim

class FNNApproximator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNNApproximator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def sarsa_fnn(model: Model, alpha: float, gamma: float, epsilon: float, episodes: int, max_steps: int, hidden_dim: int):
    num_features = model.num_states * len(Actions)
    approximator = FNNApproximator(num_features, hidden_dim, 1)
    optimizer = optim.Adam(approximator.parameters(), lr=alpha)
    criterion = nn.MSELoss()
    rewards_per_episode = []
    for episode in range(episodes):
        state = model.start_state
        action = choose_action(state, approximator(torch.FloatTensor(np.eye(num_features)[state * len(Actions) + action])).detach().numpy(), epsilon)
        total_reward = 0
        for step in range(max_steps):
            next_state = model.cell2state(model._result_action(model.state2cell(state), action))
            reward = model.reward(state, action)
            next_action = choose_action(next_state, approximator(torch.FloatTensor(np.eye(num_features)[next_state * len(Actions) + next_action])).detach().numpy(), epsilon)
            features = torch.FloatTensor(np.eye(num_features)[state * len(Actions) + action])
            next_features = torch.FloatTensor(np.eye(num_features)[next_state * len(Actions) + next_action])
            target = reward + gamma * approximator(next_features).item()
            prediction = approximator(features)
            loss = criterion(prediction, torch.tensor([target]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state, action = next_state, next_action
            total_reward += reward
            if state == model.goal_state:
                break
        rewards_per_episode.append(total_reward)
    policy = np.argmax([approximator(torch.FloatTensor(np.eye(num_features)[state * len(Actions) + action])).item() for action in range(len(Actions))] for state in range(model.num_states))
    return approximator, policy, rewards_per_episode
def create_dynamic_small_world():
    obstacle_cells = [
        Cell(1, 1),
        Cell(2, 1),
        Cell(1, 2),
    ]
    dynamic_obstacles = [cell for cell in obstacle_cells if np.random.rand() < 0.5]
    return WorldConfig(
        num_cols=4,
        num_rows=4,
        start_cell=Cell(0, 0),
        goal_cell=Cell(3, 3),
        obstacle_cells=dynamic_obstacles,
        bad_cells=[],
        prob_good_trans=0.8,
        bias=0.5,
        reward_step=-1.0,
        reward_goal=10.0,
        reward_bad=-6.0,
        gamma=0.9,
    )


class NonLinearApproximator:
    def __init__(self, num_features):
        self.weights = np.random.randn(num_features)

    def predict(self, features):
        return np.tanh(np.dot(self.weights, features))

    def update(self, features, target, alpha):
        prediction = self.predict(features)
        gradient = (1 - prediction ** 2) * features  # tanh 的导数
        self.weights += alpha * (target - prediction) * gradient

def sarsa_nonlinear(model: Model, alpha: float, gamma: float, epsilon: float, episodes: int, max_steps: int):
    num_features = model.num_states * len(Actions)
    approximator = NonLinearApproximator(num_features)
    rewards_per_episode = []
    for episode in range(episodes):
        state = model.start_state
        action = choose_action(state, approximator.weights.reshape(model.num_states, len(Actions)), epsilon)
        total_reward = 0
        for step in range(max_steps):
            next_state = model.cell2state(model._result_action(model.state2cell(state), action))
            reward = model.reward(state, action)
            next_action = choose_action(next_state, approximator.weights.reshape(model.num_states, len(Actions)), epsilon)
            features = np.zeros(num_features)
            features[state * len(Actions) + action] = 1
            next_features = np.zeros(num_features)
            next_features[next_state * len(Actions) + next_action] = 1
            target = reward + gamma * approximator.predict(next_features)
            approximator.update(features, target, alpha)
            state, action = next_state, next_action
            total_reward += reward
            if state == model.goal_state:
                break
        rewards_per_episode.append(total_reward)
    policy = np.argmax(approximator.weights.reshape(model.num_states, len(Actions)), axis=1)
    return approximator.weights, policy, rewards_per_episode
class LinearApproximator:
    def __init__(self, num_features):
        self.weights = np.zeros(num_features)

    def predict(self, features):
        return np.dot(self.weights, features)

    def update(self, features, target, alpha):
        prediction = self.predict(features)
        self.weights += alpha * (target - prediction) * features

def choose_action(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(list(Actions))
    else:
        return Actions(np.argmax(Q[state]))

def sarsa_linear(model: Model, alpha: float, gamma: float, epsilon: float, episodes: int, max_steps: int):
    num_features = model.num_states * len(Actions)
    approximator = LinearApproximator(num_features)
    rewards_per_episode = []
    for episode in range(episodes):
        state = model.start_state
        action = choose_action(state, approximator.weights.reshape(model.num_states, len(Actions)), epsilon)
        total_reward = 0
        for step in range(max_steps):
            next_state = model.cell2state(model._result_action(model.state2cell(state), action))
            reward = model.reward(state, action)
            next_action = choose_action(next_state, approximator.weights.reshape(model.num_states, len(Actions)), epsilon)
            features = np.zeros(num_features)
            features[state * len(Actions) + action] = 1
            next_features = np.zeros(num_features)
            next_features[next_state * len(Actions) + next_action] = 1
            target = reward + gamma * approximator.predict(next_features)
            approximator.update(features, target, alpha)
            state, action = next_state, next_action
            total_reward += reward
            if state == model.goal_state:
                break
        rewards_per_episode.append(total_reward)
    policy = np.argmax(approximator.weights.reshape(model.num_states, len(Actions)), axis=1)
    return approximator.weights, policy, rewards_per_episode

if __name__ == "__main__":
    alpha = 0.001  # 学习率
    gamma = 0.9  # 折扣因子
    epsilon = 0.1  # 探索参数
    episodes = 500  # 剧集数量
    max_steps = 1000  # 每个剧集的最大迭代次数
    hidden_dim = 128  # 隐藏层维度

    # 运行SARSA线性函数逼近算法
    model = Model(create_dynamic_small_world())
    weights_linear, policy_linear, rewards_sarsa_linear = sarsa_linear(model, alpha, gamma, epsilon, episodes, max_steps)
    V_sarsa_linear = np.max(weights_linear.reshape(model.num_states, len(Actions)), axis=1)
    plot_vp(model, V_sarsa_linear, policy_linear)
    plt.title("SARSA with Linear Function Approximation on Dynamic Small World")
    plt.show()

    # 运行SARSA前馈神经网络函数逼近算法
    approximator_fnn, policy_fnn, rewards_sarsa_fnn = sarsa_fnn(model, alpha, gamma, epsilon, episodes, max_steps, hidden_dim)
    V_sarsa_fnn = np.max([approximator_fnn(torch.FloatTensor(np.eye(model.num_states * len(Actions))[state * len(Actions) + action])).item() for action in range(len(Actions))] for state in range(model.num_states))
    plot_vp(model, V_sarsa_fnn, policy_fnn)
    plt.title("SARSA with FNN Function Approximation on Dynamic Small World")
    plt.show()

    # 绘制线性和非线性函数逼近的学习曲线
    plt.plot(range(1, len(rewards_sarsa_linear) + 1), rewards_sarsa_linear, label='SARSA Linear')
    plt.plot(range(1, len(rewards_sarsa_fnn) + 1), rewards_sarsa_fnn, label='SARSA FNN')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward per Episode')
    plt.title('Learning Curve for SARSA with Linear and FNN Function Approximation')
    plt.legend()
    plt.show()
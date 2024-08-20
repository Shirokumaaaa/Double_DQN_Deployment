import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import cv2

# 定义 Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# 定义 DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# 初始化环境和代理
env = gym.make('LunarLander-v2', render_mode='rgb_array')  # 改为 rgb_array 模式
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

# 训练循环
num_episodes = 1000
rewards_history = []  # 用于记录每轮游戏的总奖励

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward

        # 获取渲染图像并添加文本
        img = env.render()  # 获取当前帧
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转换颜色通道顺序
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"Episode: {episode}", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Lunar Lander", img)  # 显示图像
        cv2.waitKey(1)  # 等待一段时间，以便更新图像

    if episode % 1 == 0:
        agent.update_target_network()

    if episode >= 200:
        agent.epsilon = 0.001

    print(f"Episode: {episode}, Total Reward: {total_reward}")
    rewards_history.append(total_reward)  # 记录每轮游戏的总奖励

env.close()

# 使用 matplotlib 绘制奖励变化图
plt.figure(figsize=(10, 5))
plt.plot(rewards_history, label='Total Reward per Episode')
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.show()

cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
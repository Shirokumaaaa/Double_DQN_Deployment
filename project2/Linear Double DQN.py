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

# 定义 Double DQN Agent
class DoubleDQNAgent:
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
        self.epsilon_min = 0.1
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

        # Double Q值更新
        q_values = self.q_network(states).gather(1, actions)
        next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_network(next_states).gather(1, next_actions).detach()
        expected_q_values = rewards + self.gamma * next_q_values.squeeze() * (1 - dones)

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
agent = DoubleDQNAgent(env.observation_space.shape[0], env.action_space.n)

import pandas as pd

# 训练循环
num_episodes_per_group = 1000  # 每组运行的轮次数
num_groups = 4  # 总共训练的组数
rewards_history = [[] for _ in range(num_episodes_per_group)]  # 用于记录每轮游戏的总奖励

for group in range(num_groups):
    print(f"Training Group {group + 1}/{num_groups}")
    for episode in range(num_episodes_per_group):
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
            agent.epsilon = 0.0001

        print(f"Group: {group + 1}, Episode: {episode}, Total Reward: {total_reward}")
        rewards_history[episode].append(total_reward)  # 记录每轮游戏的总奖励

env.close()

# 计算每轮的平均奖励
average_rewards = [np.mean(rewards) for rewards in rewards_history]

# 使用 pandas 将平均奖励保存到 Excel 文件
df = pd.DataFrame(average_rewards, columns=['Average Reward'])
df.to_excel('average_rewards.xlsx', index_label='Episode')

# 使用 matplotlib 绘制奖励变化图
plt.figure(figsize=(10, 5))
plt.plot(average_rewards, label='Average Total Reward per Episode')
plt.title('Average Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Total Reward')
plt.legend()
plt.show()

cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口




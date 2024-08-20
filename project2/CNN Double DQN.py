import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义 Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(QNetwork, self).__init__()

        # 卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(0.01),  # 使用 LeakyReLU

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(0.01),  # 使用 LeakyReLU

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.01)  # 使用 LeakyReLU
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),  # 假设输入图像大小为 84x84
            nn.LeakyReLU(0.01),  # 使用 LeakyReLU
            nn.Linear(1024, output_dim)
        )


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.fc(x)
        return x

def preprocess_image(image):
    """预处理图像，缩放并转换为灰度"""
    image = cv2.resize(image, (84, 84))  # 缩放到 84x84
    if len(image.shape) == 3 and image.shape[2] == 3:  # 检查图像是否有 3 个通道
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度
    image = image / 255.0  # 归一化
    return image.reshape(1, 84, 84)  # 调整维度为 (channels, height, width)

# 定义 Double DQN Agent
class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = QNetwork(state_dim[0], action_dim).to(device)
        self.target_network = QNetwork(state_dim[0], action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        self.replay_buffer = deque(maxlen=50000)
        self.batch_size = 64
        self.gamma = 0.9
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        else:
            # 将图像转换为适合卷积神经网络的格式
            state = preprocess_image(state)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # 添加批次维度
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((preprocess_image(state), action, reward, preprocess_image(next_state), done))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 将列表转换为单一的NumPy数组
        states = np.array(states)
        next_states = np.array(next_states)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

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
agent = DoubleDQNAgent((1, 84, 84), env.action_space.n)

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

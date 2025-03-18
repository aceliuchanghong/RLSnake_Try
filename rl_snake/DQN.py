import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
import os
import numpy as np

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)
from rl_snake.ReplayBuffer import ReplayBuffer


class DQN(nn.Module):
    """
    Deep Q-Network

    - **输入**:状态是16x16的二维数组,加上通道维度后为(1, 16, 16)。
    - **卷积层**:提取游戏板的空间特征。
    - **全连接层**:将特征映射到4个动作的Q值。
    - **输出**:4个Q值,对应`UP`, `DOWN`, `LEFT`, `RIGHT`。
    """

    def __init__(self, input_shape, num_actions, dropout=0.2):
        super(DQN, self).__init__()
        # Conv2d 输入是一个四维张量 (batch_size, channels, height, width) ==> (B,C,H,W)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * input_shape[0] * input_shape[1], 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_actions)
        # self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 输入张量 x 的形状是 (1, 16, 16)，为了适应卷积层的要求，PyTorch 会自动将这个张量视为一个四维张量，其形状变为：(1, 1, 16, 16)
        x = torch.relu(self.conv1(x))  # (1, 1, 16, 16)==>(1, 32, 16, 16)
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # (B, 64 * 16 * 16) -> (B, 16384)
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # (B, num_actions)
        return x


class DQNAgent:
    """
    状态的形状、动作的数量、学习率、折扣因子、探索率、最小探索率、探索衰减率、回放缓冲区大小和batch大小
    """

    def __init__(
        self,
        state_shape,
        num_actions,
        lr=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
        buffer_size=100000,
        batch_size=256,
    ):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 策略网络 用于选择动作
        self.policy_net = DQN(state_shape, num_actions).to(self.device)
        # 目标网络 用于稳定训练, 其初始时 target_net 的权重与 policy_net 一致。
        self.target_net = DQN(state_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)

    def select_action(self, state):
        """选择动作，ε-贪婪策略"""
        if random.random() < self.epsilon:
            """
            如果随机数小于 epsilon,选择一个随机动作 探索
            否则，使用 policy_net 计算当前状态下所有动作的 Q 值，并选择 Q 值最大的动作
            """
            return random.randint(0, self.num_actions - 1)
        else:
            # 将输入的 state 转换为 Tensor (H,W)==>(1,H,W)==>(1,1,H,W)
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            # 禁用梯度计算 加速计算并节省内存
            with torch.no_grad():
                q_values = self.policy_net(state)
            # 找到 Q 值向量中最大值的索引。这个索引对应于 Q 值最高的动作 .item()将张量中的单个值转换为 Python 标量
            return q_values.argmax().item()

    def update(self):
        """更新网络 首先检查回放缓冲区中的数据量是否足够。如果足够，从回放缓冲区中采样一批数据，计算当前 Q 值和目标 Q 值，计算损失函数并进行反向传播。"""

        # 检查缓冲区大小 ：如果缓冲区中的样本不足 batch_size，直接返回
        if len(self.memory) < self.batch_size:
            return

        # 采样数据 ：从缓冲区中随机采样一批数据（状态、动作、奖励、下一状态、是否结束）。
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 将采样的数据转换为 张量
        states = torch.FloatTensor(np.array(states)).unsqueeze(1).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = (
            torch.FloatTensor(np.array(next_states)).unsqueeze(1).to(self.device)
        )
        dones = torch.FloatTensor(dones).to(self.device)

        """
        input_tensor = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])

        index_tensor = torch.tensor([
            [1],
            [2]
        ])
        
        input_tensor : 形状为 (2, 3)
        index_tensor : 形状为 (2, 1)
        
        input_tensor.gather(1, index_tensor)
        
        output = torch.tensor([
            [2.0],
            [6.0]
        ])
        
        对于每个 batch 即每一行，根据 index_tensor 中的索引收集 input_tensor 中的元素
        沿着第二个维度 dim=1 收集 Q 值
        index_tensor[0][0]
        index_tensor[1][0]
        """
        current_q_values = (
            # states = (B,) ==> policy_net = (B, num_actions) ==> actions=4, unsqueeze = (4, 1)
            # ==> gather 沿着第二个维度（dim=1）收集 Q 值 ==> squeeze 移除第二个维度 转换为形状为 (B,)
            self.policy_net(states)
            .gather(1, actions.unsqueeze(1))
            .squeeze(1)
        )
        with torch.no_grad():
            # 使用 target_net 计算下一状态的最大 Q 值
            max_next_q_values = self.target_net(next_states).max(1)[0]
            # 计算目标 Q 值
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_net(self):
        """更新目标网络 使其与策略网络的参数一致"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

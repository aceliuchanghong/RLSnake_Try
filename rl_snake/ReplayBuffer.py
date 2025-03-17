from collections import deque
import random


class ReplayBuffer:
    """
    DQN需要存储过去的经验 状态、动作、奖励、下一状态 ==> state, action, reward, next_state, 从中采样训练。

    d = deque([1, 2, 3])

    d.append(4)         # 从右侧添加
    print(d)            # 输出: deque([1, 2, 3, 4])
    d.appendleft(0)     # 从左侧添加
    print(d)            # 输出: deque([0, 1, 2, 3, 4])
    """

    def __init__(self, capacity):
        # 参数 capacity，表示回放缓冲区的最大容量。
        # 双端队列，设置最大长度为 capacity。这个队列将用于存储经验元组。
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 从回放缓冲区中随机选择 batch_size 个经验元组，并返回这些元组。
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

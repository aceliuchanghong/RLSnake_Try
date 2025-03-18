import os
import sys


sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)
from snake.SnakeGameMain import SnakeGame


class SnakeEnv(SnakeGame):
    def __init__(self, width=16, height=16, show=False):
        super().__init__(width, height, show)
        self.prev_distance = None  # 上一步蛇头到食物的距离
        self.max_steps = width * height - 1

    def reset(self):
        """重置游戏，返回初始状态"""
        self.snake = [(self.width // 2, self.height // 2)]
        self.food = self._generate_food()
        self.direction = "UP"
        self.game_over = False
        self.steps = 0
        self.score = 0
        self.current_steps = 0
        self.prev_distance = self._calculate_distance(self.snake[0], self.food)
        return self.get_state()

    def _calculate_distance(self, pos1, pos2):
        """计算距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self, action):
        """执行一步动作,返回其执行完之后的状态、奖励和是否结束"""
        # 各个奖励项的权重（例如 0.1、0.01）需要通过实验调整，确保它们不会相互抵消或导致学习不稳定

        super().step(action)
        self.current_steps += 1
        reward = -0.05  # 每步小惩罚，鼓励快速吃到食物

        if self.game_over:
            # 简化游戏结束的惩罚，设置为固定值
            reward = -10
        else:
            if self.snake[0] == self.food:
                # 显著增加吃到食物的奖励
                reward = 20
                self.prev_distance = None  # 重置距离，因为食物位置会改变
            else:
                # 调整接近或远离食物的奖励权重
                current_distance = self._calculate_distance(self.snake[0], self.food)
                if self.prev_distance is not None:
                    if current_distance < self.prev_distance:
                        reward += 1.0  # 增加接近食物的奖励
                    elif current_distance > self.prev_distance:
                        reward -= 0.8  # 增加远离食物的惩罚
                    else:
                        reward -= 0.2  # 距离不变时的惩罚
                self.prev_distance = current_distance

        # 防止无限循环
        if self.current_steps >= self.max_steps:
            self.game_over = True
            reward = -20  # 降低循环的惩罚，但仍保持负值

        return self.get_state(), reward, self.game_over

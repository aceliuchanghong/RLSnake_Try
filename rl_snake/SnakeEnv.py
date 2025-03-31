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
        self.max_steps = width * height * 2 - 1

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
        prev_length = len(self.snake)
        super().step(action)
        reward = -0.1  # 每步小惩罚，鼓励快速行动
        if self.game_over:
            # reward = -10 - 0.5 * (len(self.snake) - 1)
            reward = -20  # 游戏结束惩罚
        else:
            if len(self.snake) > prev_length:
                reward = 20 + 0.5 * (len(self.snake) - 1)  # 吃食物奖励
                self.current_steps = 0
                self.prev_distance = None
            else:
                current_distance = self._calculate_distance(self.snake[0], self.food)
                if self.prev_distance is not None:
                    if current_distance < self.prev_distance:
                        reward += 0.3  # 接近奖励
                    elif current_distance > self.prev_distance:
                        reward -= 0.32  # 远离惩罚
                self.prev_distance = current_distance
                self.current_steps += 1
        if self.current_steps >= self.max_steps:
            self.game_over = True
            reward = -20
        if len(self.snake) > (self.width * self.height - 20):
            reward += 150

        return self.get_state(), reward, self.steps, self.game_over

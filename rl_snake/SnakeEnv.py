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
        super().step(action)
        reward = -0.01
        if self.game_over:
            reward = -20
        else:
            if self.snake[0] == self.food:
                reward = 20 + 2 * (len(self.snake) - 1)  # 增加吃食物奖励
                self.current_steps = 0
                self.prev_distance = None
            else:
                current_distance = self._calculate_distance(self.snake[0], self.food)
                if self.prev_distance is not None:
                    if current_distance < self.prev_distance:
                        reward += 0.5  # 接近奖励
                    elif current_distance > self.prev_distance:
                        reward -= 0.5  # 远离惩罚
                    else:
                        reward -= 0.1  # 不变惩罚
                self.prev_distance = current_distance
                self.current_steps += 1
        if self.current_steps >= self.max_steps:
            self.game_over = True
            reward = -40
        return self.get_state(), reward, self.steps, self.game_over

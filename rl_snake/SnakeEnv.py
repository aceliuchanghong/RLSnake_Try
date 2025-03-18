import os
import sys
import numpy as np

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)
from snake.SnakeGameMain import SnakeGame


class SnakeEnv(SnakeGame):
    def reset(self):
        """重置游戏，返回初始状态"""
        self.snake = [(self.width // 2, self.height // 2)]
        self.food = self._generate_food()
        self.direction = "UP"
        self.game_over = False
        self.steps = 0
        self.score = 0
        return self.get_state()

    def get_state(self):
        """获取当前状态，用二维数组表示游戏板"""
        state = np.zeros((self.height, self.width), dtype=np.float32)
        for x, y in self.snake:
            state[y, x] = 1.0  # 蛇身标记为1
        state[self.snake[0][1], self.snake[0][0]] = 0.5  # 蛇头标记为0.5
        state[self.food[1], self.food[0]] = -1.0  # 食物标记为-1
        return state

    def step(self, action):
        """执行一步动作，返回状态、奖励和是否结束"""
        super().step(action)  # 调用父类的step方法
        reward = 0
        if self.game_over:
            reward = -1  # 撞墙或撞自己，负奖励
        elif self.snake[0] == self.food:
            reward = 1  # 吃到食物，正奖励
        else:
            reward = -0.01  # 每步小负奖励，鼓励快速吃到食物
        return self.get_state(), reward, self.game_over

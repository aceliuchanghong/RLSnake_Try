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

    def reset(self):
        """重置游戏，返回初始状态"""
        self.snake = [(self.width // 2, self.height // 2)]
        self.food = self._generate_food()
        self.direction = "UP"
        self.game_over = False
        self.steps = 0
        self.score = 0
        self.prev_distance = self._calculate_distance(self.snake[0], self.food)
        return self.get_state()

    def _calculate_distance(self, pos1, pos2):
        """计算距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self, action):
        """执行一步动作,返回其执行完之后的状态、奖励和是否结束"""
        # 各个奖励项的权重（例如 0.1、0.01）需要通过实验调整，确保它们不会相互抵消或导致学习不稳定

        super().step(action)  # 调用父类的step方法
        reward = 0
        if self.game_over:
            # 输掉游戏的惩罚，与蛇的长度成反比
            reward = 5 - 10 / (1 + 0.1 * (len(self.snake) - 1))
        else:
            if self.snake[0] == self.food:
                # 吃到食物的奖励，与蛇的长度成正比
                reward += 5 + 0.3 * (len(self.snake) - 1)
                self.prev_distance = None  # 重置距离，因为食物位置会改变
            else:
                # 接近或远离食物的奖励
                current_distance = self._calculate_distance(self.snake[0], self.food)
                if self.prev_distance is not None:
                    if current_distance < self.prev_distance:
                        reward += 0.3  # 接近食物
                    elif current_distance > self.prev_distance:
                        reward -= 0.3  # 远离食物
                    else:
                        reward -= 0.1  # 距离不变
                self.prev_distance = current_distance

                # 鼓励蛇头远离蛇尾
                if len(self.snake) > 1:
                    tail_distance = self._calculate_distance(
                        self.snake[0], self.snake[-1]
                    )
                    reward += 0.01 * tail_distance
        return self.get_state(), reward, self.game_over

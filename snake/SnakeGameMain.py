import random
import os
import time
from termcolor import colored
import numpy as np


class SnakeGame:
    def __init__(self, width=16, height=16, show=False):
        self.width = width
        self.height = height
        self.snake = [(width // 2, height // 2)]
        self.food = self._generate_food()
        self.direction = "UP"
        self.game_over = False
        self.show = show
        self.steps = 0
        self.score = 0
        self.current_steps = 0

    def _generate_food(self):
        """随机生成食物位置，确保不在蛇身上"""
        while True:
            food = (
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1),
            )
            if food not in self.snake:
                return food

    def step(self, action):
        """执行一步动作，返回游戏状态"""
        if self.game_over:
            return
        self.steps += 1

        # 根据动作更新方向（避免反向移动）
        if action == "UP" and self.direction != "DOWN":
            self.direction = "UP"
        elif action == "DOWN" and self.direction != "UP":
            self.direction = "DOWN"
        elif action == "LEFT" and self.direction != "RIGHT":
            self.direction = "LEFT"
        elif action == "RIGHT" and self.direction != "LEFT":
            self.direction = "RIGHT"

        # 计算新头部位置
        head = self.snake[0]
        if self.direction == "UP":
            new_head = (head[0], head[1] - 1)
        elif self.direction == "DOWN":
            new_head = (head[0], head[1] + 1)
        elif self.direction == "LEFT":
            new_head = (head[0] - 1, head[1])
        elif self.direction == "RIGHT":
            new_head = (head[0] + 1, head[1])

        # 检查碰撞（墙壁或自身）
        if (
            new_head in self.snake
            or new_head[0] < 0
            or new_head[0] >= self.width
            or new_head[1] < 0
            or new_head[1] >= self.height
        ):
            self.game_over = True
            return

        # 移动蛇
        self.snake.insert(0, new_head)
        if new_head == self.food:
            # 吃到食物，生成新食物，不缩短尾巴，增加得分
            self.food = self._generate_food()
            self.score += 1  # 增加得分
        else:
            # 未吃到食物，尾巴缩短
            self.snake.pop()

    def get_state(self):
        """获取当前状态，用二维数组表示游戏板"""
        state = np.zeros((self.height, self.width), dtype=np.float32)
        for x, y in self.snake:
            state[y, x] = 1.0  # 蛇身标记为1
        state[self.snake[0][1], self.snake[0][0]] = 0.5  # 蛇头标记为0.5
        state[self.food[1], self.food[0]] = -1.0  # 食物标记为-1
        return state

    def render(self, speed=0.1, other_info={}):
        """在命令行中渲染游戏状态"""
        if not self.show:
            return

        os.system("clear" if os.name == "posix" else "cls")
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.snake:
                    if (x, y) == self.snake[0]:
                        print(colored("H", "green"), end=" ")
                    else:
                        print(colored("S", "yellow"), end=" ")
                elif (x, y) == self.food:
                    print(colored("F", "red"), end=" ")
                else:
                    print(".", end=" ")
            print()

        info_str = (
            f"Steps: {self.steps}, Score: {self.score}, Direction: {self.direction}"
        )
        if other_info:
            info_str += ", " + ", ".join(
                f"{str(key)}: {str(value)}" for key, value in other_info.items()
            )
        print(colored(info_str, "cyan"))
        time.sleep(speed)

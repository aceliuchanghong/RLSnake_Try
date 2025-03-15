import os
import sys
import random

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)
from snake.SnakeGameMain import SnakeGame

if __name__ == "__main__":
    """
    uv run snake/run.py
    """
    game = SnakeGame(show=True)
    actions = ["UP", "DOWN", "LEFT", "RIGHT"]
    while not game.game_over:
        action = random.choice(actions)
        game.step(action)
        game.render()

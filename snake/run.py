import os
from dotenv import load_dotenv
import logging
from termcolor import colored
import sys
import random

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
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

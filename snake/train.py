import torch
import os
from dotenv import load_dotenv
import logging
from termcolor import colored
import csv
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)
from rl_snake.SnakeEnv import SnakeEnv
from rl_snake.DQN import DQNAgent

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    """
    uv run snake/train.py
    nohup uv run snake/train.py > no_git_oic/train.log 2>&1 &
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SnakeEnv(width=16, height=16)
    state_shape = (16, 16)
    num_actions = 4

    agent = DQNAgent(state_shape, num_actions)
    actions = ["UP", "DOWN", "LEFT", "RIGHT"]

    num_epochs = 200000
    target_update_freq = 200
    target_save_freq = 2000
    update_freq = 20  # 每20步更新一次网络
    step_counter = 0  # 全局步数计数器
    max_foods = 0
    file_path = "no_git_oic/train.csv"

    from collections import deque

    # 使用一个队列来存储最近 target_update_freq 次的奖励
    recent_rewards = deque(maxlen=target_update_freq)

    for epoch in range(num_epochs):
        state = env.reset()
        epoch_reward = 0
        done = False
        while not done:
            action_idx = agent.select_action(state)
            action = actions[action_idx]
            next_state, reward, steps, done = env.step(action)
            agent.memory.push(state, action_idx, reward, next_state, done)
            state = next_state
            epoch_reward += reward
            step_counter += 1
            if step_counter % update_freq == 0:
                agent.update()

        recent_rewards.append(epoch_reward)
        max_foods = env.score if env.score > max_foods else max_foods

        if (epoch + 1) % target_update_freq == 0:
            agent.update_target_net()

            avg_reward = sum(recent_rewards) / len(recent_rewards)
            file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0
            with open(file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["Epochs", "Average_Reward"])
                writer.writerow([epoch + 1, avg_reward])
            logger.info(
                colored(
                    f"\n-------Average Reward: {avg_reward:.2f}-------"
                    + f"\nEpochs {(epoch+1):5}, Rewards: {epoch_reward:.2f}, Foods: {env.score:3}-{max_foods},"
                    + (
                        f" Epsilon: {agent.epsilon:.2f},"
                        if agent.epsilon >= agent.epsilon_min
                        else ""
                    )
                    + f" Steps: {steps:5}",
                    "green",
                )
            )
        if (epoch + 1) % target_save_freq == 0:
            torch.save(
                agent.policy_net.state_dict(),
                f"rl_model/dqn_snake_best_{str(epoch+1)}.pth",
            )

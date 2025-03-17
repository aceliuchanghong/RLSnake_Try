import numpy as np
import torch
import torch.nn as nn

from rl_snake.SnakeEnv import SnakeEnv
from rl_snake.DQN import DQNAgent

if __name__ == "__main__":
    """
    uv run infer.py
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SnakeEnv(width=16, height=16, show=True)
    state_shape = (16, 16)
    num_actions = 4

    agent = DQNAgent(state_shape, num_actions)
    agent.policy_net.load_state_dict(torch.load("dqn_snake.pth"))
    agent.policy_net.eval()

    state = env.reset()
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = agent.policy_net(state_tensor)
        action_idx = q_values.argmax().item()
        action = ["UP", "DOWN", "LEFT", "RIGHT"][action_idx]
        state, reward, done = env.step(action)
        env.render(speed=0.1)

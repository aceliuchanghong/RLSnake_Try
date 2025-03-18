import numpy as np
import torch
import torch.nn as nn

from rl_snake.SnakeEnv import SnakeEnv
from rl_snake.DQN import DQNAgent

if __name__ == "__main__":
    """
    uv run train.py
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SnakeEnv(width=16, height=16, show=False)
    state_shape = (16, 16)
    num_actions = 4

    agent = DQNAgent(state_shape, num_actions, batch_size=128)
    actions = ["UP", "DOWN", "LEFT", "RIGHT"]

    num_episodes = 1000
    target_update_freq = 10

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action_idx = agent.select_action(state)
            action = actions[action_idx]
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.update()

        if episode % target_update_freq == 0:
            agent.update_target_net()

        print(
            f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}"
        )

    torch.save(agent.policy_net.state_dict(), "dqn_snake.pth")

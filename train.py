import torch
from rl_snake.SnakeEnv import SnakeEnv
from rl_snake.DQN import DQNAgent

if __name__ == "__main__":
    """
    uv run train.py
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SnakeEnv(width=16, height=16)
    state_shape = (16, 16)
    num_actions = 4

    agent = DQNAgent(state_shape, num_actions)
    actions = ["UP", "DOWN", "LEFT", "RIGHT"]

    num_epochs = 10000
    target_update_freq = 100
    target_save_freq = 1000
    max_foods = 0

    for epoch in range(num_epochs):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action_idx = agent.select_action(state)
            action = actions[action_idx]
            next_state, reward, steps, done = env.step(action)
            agent.memory.push(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.update()

        if (epoch + 1) % target_update_freq == 0:
            agent.update_target_net()
        if (epoch + 1) % target_save_freq == 0:
            torch.save(
                agent.policy_net.state_dict(), f"dqn_snake_best_{str(epoch+1)}.pth"
            )
        max_foods = env.score if env.score > max_foods else max_foods
        print(
            f"Epochs {epoch:5}, Rewards: {total_reward:.2f}, Foods: {env.score:3}-{max_foods},"
            + (
                f" Epsilon: {agent.epsilon:.2f},"
                if agent.epsilon >= agent.epsilon_min
                else ""
            )
            + f" Steps: {steps:5}"
        )

import torch
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

    agent = DQNAgent(state_shape, num_actions)
    actions = ["UP", "DOWN", "LEFT", "RIGHT"]

    num_epochs = 10000
    target_update_freq = 1000

    for epoch in range(num_epochs):
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

        if (epoch + 1) % target_update_freq == 0:
            agent.update_target_net()
            torch.save(
                agent.policy_net.state_dict(), f"dqn_snake_best_{str(epoch+1)}.pth"
            )
        print(f"Epoch {epoch}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

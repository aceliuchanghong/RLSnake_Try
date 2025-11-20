import torch
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)
from rl_snake.SnakeEnv import SnakeEnv
from rl_snake.DQN import DQNAgent

if __name__ == "__main__":
    """
    uv run snake/infer.py

    cpu 版本:
        - uv pip uninstall torch
        - uv pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-cache-dir
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    game = SnakeEnv(show=True)

    state_shape = (16, 16)
    num_actions = 4
    agent = DQNAgent(state_shape, num_actions)
    agent.policy_net.load_state_dict(
        torch.load("rl_model/dqn_snake_best.pth", map_location=device)
    )
    agent.policy_net.to(device)
    agent.policy_net.eval()
    total_reward = 0

    while not game.game_over:
        state = game.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = agent.policy_net(state_tensor)
        action_idx = q_values.argmax().item()
        action = ["UP", "DOWN", "LEFT", "RIGHT"][action_idx]
        _, reward_now, _, _ = game.step(action)
        total_reward += reward_now
        game.render(speed=0.1, other_info={"Reward": total_reward})

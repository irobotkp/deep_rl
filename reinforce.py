"""
Reinforce: policy gradient alg. implementation
# TODO: make it device agnostic
"""
from pathlib import Path
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np
import gymnasium as gym


#------------- Arguments --------------------------
parser = ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--eval', action='store_true')
args = parser.parse_args()

#------------- Hyperparameters --------------------
ENV_NAME = "Acrobot-v1"
LR = 1e-3
GAMMA = 0.99
EPISODES = 500
SAVE_PATH = './checkpoints/reinforce.pth'
#--------------------------------------------------

class PolicyNetwork(nn.Module):
    """policy: outputs action logits"""
    def __init__(self, n_obs: int, n_act: int, n_hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_act)
        )

    def forward(self, x: Tensor) -> Tensor:
        """returns action logits"""
        return self.net(x)

class ReinforceAgent:
    """reinforce agent class"""
    def __init__(self, n_obs: int, n_act: int) -> None:
        self.policy = PolicyNetwork(n_obs, n_act, n_hidden=128)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=LR)

    def select_action(self, state: np.ndarray) -> tuple[float, Tensor]:
        """returns: action and log prob"""
        logits = self.policy(torch.from_numpy(state).unsqueeze(dim=0))
        probs = nn.functional.softmax(logits, dim=1)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def update_policy(self, log_probs: list[Tensor], returns: Tensor) -> None:
        """policy gradient update"""
        loss = sum((-log_prob * r for log_prob, r in zip(log_probs, returns)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_policy(self) -> None:
        """save current policy"""
        path = Path(SAVE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), f=path)
        print(f"model saved at {path}")

    def load_policy(self) -> None:
        """load policy from save path"""
        path = Path(SAVE_PATH)
        if path.exists():
            self.policy.load_state_dict(torch.load(f=path))
            print(f"loaded policy from {path}")
        else:
            print(f"no model found at {path}")

def compute_returns(rewards: list[float]) -> Tensor:
    """compute discounted return given rewards"""
    returns: list[float] = []
    G = 0.0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    return_tensor = torch.tensor(returns, dtype=torch.float32)
    return return_tensor

def run_train_loop() -> None:
    """run main reinfoce agent training loop"""
    env = gym.make(ENV_NAME)
    n_obs = env.observation_space.shape[0]
    n_act = env.action_space.n
    agent = ReinforceAgent(n_obs, n_act)
    agent.load_policy()

    for episode in range(EPISODES):

        state, _ = env.reset()
        done, truncated = False, False

        # store log_probs and rewards during rollout
        log_probs: list[Tensor] = []
        rewards: list[float] = []

        while not done and not truncated:

            action, log_prob = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

        returns = compute_returns(rewards)
        agent.update_policy(log_probs, returns)

        total_reward = sum(rewards)
        if episode % 20 == 0:
            print(f"Episode: {episode}, Total reward: {total_reward}")
    
    agent.save_policy()

def eval_loop() -> None:
    """eval loop"""
    env = gym.make(ENV_NAME, render_mode="human")
    n_obs = env.observation_space.shape[0]
    n_act = env.action_space.n
    agent = ReinforceAgent(n_obs, n_act)
    agent.load_policy()

    state, _ = env.reset()
    done, truncated = False, False
    total_reward = 0.0

    while not done and not truncated:

        action, log_prob = agent.select_action(state)
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward

    print(f"Evaluation: total reward: {total_reward}")

def main():
    """main training loop"""

    if args.train: run_train_loop()
    if args.eval: eval_loop()

if __name__ == "__main__":
    main()

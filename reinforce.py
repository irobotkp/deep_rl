"""
Reinforce: policy gradient alg. implementation
"""

from pathlib import Path
from argparse import ArgumentParser
from typing import Union
import random

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np
import gymnasium as gym

from utils import set_seed, get_device

# ------------- Arguments --------------------------
parser = ArgumentParser()
parser.add_argument("--train", action="store_true", help="run training")
parser.add_argument("--eval", action="store_true", help="run evaluation")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--env", type=str, default="CartPole-v1", help="environment name")
parser.add_argument(
    "--episodes", type=int, default=500, help="number of episodes to train"
)
args = parser.parse_args()

# ------------- Hyperparameters --------------------
ENV_NAME = args.env
LR = args.lr
GAMMA = 0.99
EPISODES = args.episodes
SAVE_PATH = "./checkpoints/"
DEVICE = "cpu"  # get_device()
EPS_START = 0.0
EPS_END = 0.0
# --------------------------------------------------


class DescretePolicyNetwork(nn.Module):
    """policy: outputs action logits"""

    def __init__(self, n_obs: int, n_act: int, n_hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden * 2),
            nn.ReLU(),
            nn.Linear(n_hidden * 2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_act),
        )

    def forward(self, x: Tensor) -> Tensor:
        """returns action logits"""
        return self.net(x)


class ContinuousPolicyNetwork(nn.Module):
    """policy: outputs mean and std for actions"""

    def __init__(self, n_obs: int, n_act: int, n_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden * 2),
            nn.ReLU(),
            nn.Linear(n_hidden * 2, n_hidden),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(n_hidden, n_act)
        self.std_head = nn.Linear(n_hidden, n_act)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """returns mean and log_std for actions"""
        x = self.net(x)
        log_std = torch.nn.functional.softplus(self.std_head(x))
        log_std = log_std.clamp(min=1e-3, max=1.0)
        return self.mu_head(x), log_std


class ReinforceAgent:
    """reinforce agent class"""

    def __init__(self, n_obs: int, n_act: int, is_continuous: bool) -> None:
        self.is_continuous = is_continuous
        self.n_act = n_act

        if self.is_continuous:
            self.policy = ContinuousPolicyNetwork(n_obs, n_act)
        else:
            self.policy = DescretePolicyNetwork(n_obs, n_act)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR)

        self.policy = self.policy.to(DEVICE)

    def select_action(
        self, state: np.ndarray, noise: float = 0.0
    ) -> tuple[Union[float, np.ndarray], Tensor]:
        """wrapper for discrete and continuous select action methods"""
        if self.is_continuous:
            return self.select_contiuous_action(state, noise)
        return self.select_descrete_action(state, noise)

    def select_descrete_action(self, state: np.ndarray, noise: float) -> tuple[float, Tensor]:
        """returns: action and log prob"""
        logits = self.policy(torch.from_numpy(state).unsqueeze(dim=0).to(DEVICE))
        probs = nn.functional.softmax(logits, dim=1)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        if random.random() < noise:
            action = torch.randint(0, self.n_act, (1,))
        return action.detach().cpu().item(), m.log_prob(action)

    def select_contiuous_action(
        self, state: np.ndarray, noise: float = 0.0
    ) -> tuple[Union[float, np.ndarray], Tensor]:
        """returns: action and log prob"""
        if state.ndim > 1:
            state = state.flatten()
        mu, log_std = self.policy(torch.from_numpy(state).unsqueeze(dim=0).to(DEVICE))
        std = torch.exp(log_std)
        m = torch.distributions.Normal(mu, std)
        action = m.sample()
        action += torch.rand_like(action) * noise
        action = torch.nn.functional.tanh(action)
        log_prob = m.log_prob(action).sum(dim=-1)
        return action.detach().cpu().numpy().flatten(), log_prob

    def update_policy(self, log_probs: list[Tensor], returns: Tensor) -> None:
        """policy gradient update"""
        loss = sum((-log_prob * r for log_prob, r in zip(log_probs, returns)))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

    def save_policy(self) -> None:
        """save current policy"""
        file_name = f"reinforce_{ENV_NAME}.pth"
        path = Path(SAVE_PATH) / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), f=path)
        print(f"model saved at {path}")

    def load_policy(self) -> None:
        """load policy from save path"""
        file_name = f"reinforce_{ENV_NAME}.pth"
        path = Path(SAVE_PATH) / file_name
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
    returns = np.array(returns)
    return_tensor = torch.from_numpy(returns).type(torch.float32).to(DEVICE)
    return_tensor = (return_tensor - return_tensor.mean()) / (
        return_tensor.std() + 1e-9
    )
    return return_tensor


def get_env_info(env: gym.Env) -> tuple[int, int, bool]:
    """returns n_obs, n_act and if env is continuous or not"""
    n_obs = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        is_continuous = False
        n_act = int(env.action_space.n)
    else:
        is_continuous = True
        n_act = env.action_space.shape[0]
    return n_obs, n_act, is_continuous


def run_train_loop() -> None:
    """run main reinfoce agent training loop"""
    env = gym.make(ENV_NAME)
    n_obs, n_act, is_continuous = get_env_info(env)
    agent = ReinforceAgent(n_obs, n_act, is_continuous)
    agent.load_policy()
    agent.policy.train()
    epsilon = EPS_START

    for episode in range(EPISODES):

        state, _ = env.reset()
        done, truncated = False, False

        # store log_probs and rewards during rollout
        log_probs: list[Tensor] = []
        rewards: list[float] = []

        while not done and not truncated:

            action, log_prob = agent.select_action(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

        returns = compute_returns(rewards)
        agent.update_policy(log_probs, returns)

        total_reward = sum(rewards)
        if episode % 20 == 0:
            print(
                f"Episode: {episode}, Total reward: {total_reward}, epsilon: {epsilon}"
            )

        epsilon = max(
            EPS_END, EPS_START - ((EPS_START - EPS_END) * (episode / (EPISODES * 0.5)))
        )

    agent.save_policy()


@torch.no_grad()
def eval_loop() -> None:
    """eval loop"""
    env = gym.make(ENV_NAME, render_mode="human")
    n_obs, n_act, is_continuous = get_env_info(env)
    agent = ReinforceAgent(n_obs, n_act, is_continuous)
    agent.load_policy()
    agent.policy.eval()

    with torch.inference_mode():

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

    set_seed(42)

    if args.train:
        run_train_loop()
    if args.eval:
        eval_loop()


if __name__ == "__main__":
    main()

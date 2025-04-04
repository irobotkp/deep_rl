"""
Deep Q Network implementation
"""

from typing import Tuple
from collections import namedtuple, deque
import random
from pathlib import Path
import argparse

import numpy as np

import torch
from torch import nn
from torch import Tensor

import gymnasium as gym
import matplotlib.pyplot as plt

# --------------- Arguments -------------------
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--eval', action='store_true')
args = parser.parse_args()

# --------------- Hyperparameters -------------
BATCH_SIZE = 32
LR = 1e-3
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.99
EPISODES = 200
TARGET_UPDATE_FREQ = 20
BUFFER_CAPACITY = 10000
SAVE_PATH = "./checkpoints/models/dqn.pth"
# ---------------------------------------------


Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """class to manage experiences"""

    def __init__(self, capacity: int) -> None:
        self.memory: deque = deque([], maxlen=capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, *args) -> None:
        """
        push experience in memory
        *args: state, action, reward, next_state, done
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        sample transitions of batch size from buffer
        """
        transitions = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )


class QNetwork(nn.Module):
    """Q network: state -> action"""

    def __init__(self, n_obs: int, n_action: int, n_hidden: int):
        super().__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(n_obs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_action),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: intput state
        returns: logits size of n_action
        """
        return self.linear_block(x)


class DQNAgent:
    """Q Agent"""

    def __init__(self, n_obs: int, n_action: int) -> None:

        self.n_action = n_action

        self.buffer = ReplayBuffer(BUFFER_CAPACITY)
        self.q_network = QNetwork(n_obs, n_action, n_hidden=128)
        self.target_network = QNetwork(n_obs, n_action, n_hidden=128)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.q_network.parameters(), lr=LR)
        self.steps_done = 0

    def update_target_network(self) -> None:
        """udpate target network from q network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """select action based on epsilong greedy policy"""
        if random.random() < epsilon:
            return random.randint(0, self.n_action - 1)
        else:
            return self.q_network(torch.from_numpy(state)).argmax().item()

    def train_step(self):
        """main training loop"""
        if len(self.buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        with torch.no_grad():
            target_q_values = rewards + GAMMA * self.target_network(next_states).max(
                dim=1
            )[0] * (1 - dones)
            target_q_values = target_q_values.unsqueeze(dim=1)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(dim=1))
        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.update_target_network()

    def save_model(self) -> None:
        """save q network"""
        path = Path(SAVE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_network.state_dict(), path)
        print(f"model save to {SAVE_PATH}")

    def load_model(self) -> None:
        """load model if available"""
        path = Path(SAVE_PATH)
        if path.exists():
            self.q_network.load_state_dict(torch.load(f=path))
            self.update_target_network()
            print(f"model loaded from {SAVE_PATH}")
        else:
            print(f"model not found at {SAVE_PATH}")


def plot_data(history: list[float]) -> None:
    """Plot data"""
    plt.plot(history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()


def train_q_agent(env: gym.Env) -> None:
    """trains agent given env"""
    n_obs = env.observation_space.shape[0]
    n_action = env.action_space.n

    agent = DQNAgent(n_obs, n_action)

    # load model
    agent.load_model()

    epsilon = EPS_START
    reward_history = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:

            action = agent.select_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            agent.train_step()

        reward_history.append(total_reward)

        epsilon = max(
            EPS_END, EPS_START - (episode / (EPISODES * 0.5)) * (EPS_START - EPS_END)
        )
        if episode % 20 == 0:
            print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {epsilon}")

    # save model
    agent.save_model()

    # plot
    plot_data(reward_history)

def eval_q_agent(env: gym.Env) -> None:
    """eval trained agent"""
    n_obs = env.observation_space.shape[0]
    n_action = env.action_space.n

    agent = DQNAgent(n_obs, n_action)
    agent.load_model()

    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:

        action = agent.select_action(state, epsilon=0.0)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward

    print(f"Total reward: {total_reward}")


def main() -> None:
    """Run agent"""

    if args.train:
        env = gym.make("CartPole-v1", render_mode=None)
        train_q_agent(env)

    if args.eval:
        env = gym.make("CartPole-v1", render_mode="human")
        eval_q_agent(env)


if __name__ == "__main__":
    main()


"""
Deep Q-Learning conversion of the grid-world Q-learning example.

This implementation uses PyTorch (CPU) and trains a small MLP to predict Q-values
for the 4 actions given a one-hot encoding of the 12 states.

If PyTorch is not installed, install it with: pip install torch
"""

import random
from collections import deque
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim



#environement parameters
n_states = 12
n_actions = 4
goal_state = (0, 3)
reward_goal = 1.0
penalty_state = (1, 3)
penalty = -1.0
step_cost = -0.04

rows, columns = 3, 4
all_states = {(r, c) for r in range(rows) for c in range(columns)}
all_actions = ["N", "E", "S", "W"]

blocked_cell = (1, 1)

left_action = {"N": "W", "E": "N", "S": "E", "W": "S"}
right_action = {"N": "E", "E": "S", "S": "W", "W": "N"}


def move(state, action):
    r, c = state
    if action == "N":
        r2, c2 = max(r - 1, 0), c
    elif action == "E":
        r2, c2 = r, min(c + 1, columns - 1)
    elif action == "S":
        r2, c2 = min(r + 1, rows - 1), c
    elif action == "W":
        r2, c2 = r, max(c - 1, 0)
    s2 = (r2, c2)
    if s2 == blocked_cell:
        s2 = state
    return s2


transition_probs = {}
for state in all_states:
    for action in all_actions:
        if state == goal_state or state == penalty_state:
            transition_probs[(state, action)] = [(state, 1.0)]
            continue
        intended = move(state, action)
        left = move(state, left_action[action])
        right = move(state, right_action[action])
        transition_probs[(state, action)] = [(intended, 0.8), (left, 0.1), (right, 0.1)]


# helper maps
state_to_index = lambda s: s[0] * columns + s[1]


# --- DQN components ---
class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def one_hot_state(state_index: int, n: int = n_states):
    vec = np.zeros(n, dtype=np.float32)
    vec[state_index] = 1.0
    return vec


def select_action(policy_net, state_vec, epsilon):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        s = torch.from_numpy(state_vec).unsqueeze(0)
        qvals = policy_net(s)
        return int(torch.argmax(qvals, dim=1).item())


def train_step(policy_net, target_net, optimizer, buffer, batch_size, gamma, device):
    if len(buffer) < batch_size:
        return None
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    states_v = torch.from_numpy(states).to(device)
    next_states_v = torch.from_numpy(next_states).to(device)
    actions_v = torch.from_numpy(actions).long().to(device)
    rewards_v = torch.from_numpy(rewards).float().to(device)
    dones_v = torch.from_numpy(dones.astype(np.uint8)).float().to(device)

    q_values = policy_net(states_v).gather(1, actions_v.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = target_net(next_states_v).max(1)[0]
        target_q = rewards_v + gamma * next_q_values * (1.0 - dones_v)

    loss = nn.MSELoss()(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def run_dqn(
    episodes=1000,
    batch_size=32,
    gamma=0.99,
    lr=1e-3,
    buffer_capacity=5000,
    target_update=50,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=500,
    device=None,
):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    input_dim = n_states
    policy_net = DQN(input_dim, n_actions).to(device)
    target_net = DQN(input_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)

    episode_rewards = []

    for ep in range(1, episodes + 1):
        current_state = (2, 0)
        state_idx = state_to_index(current_state)
        state_vec = one_hot_state(state_idx)
        total_reward = 0.0
        steps = 0

        done = False
        while not done and steps < 100:
            steps += 1
            # linearly decaying epsilon
            eps = eps_end + (eps_start - eps_end) * math.exp(-1.0 * ep / eps_decay)
            action_idx = select_action(policy_net, state_vec, eps)
            action = all_actions[action_idx]

            next_states, probs = zip(*transition_probs[(current_state, action)])
            next_state = random.choices(next_states, probs)[0]

            if next_state == goal_state:
                reward = reward_goal
                done = True
            elif next_state == penalty_state:
                reward = penalty
                done = True
            else:
                reward = step_cost

            next_state_idx = state_to_index(next_state)
            next_state_vec = one_hot_state(next_state_idx)

            buffer.push(state_vec, action_idx, reward, next_state_vec, done)

            loss = train_step(policy_net, target_net, optimizer, buffer, batch_size, gamma, device)

            total_reward += reward
            current_state = next_state
            state_vec = next_state_vec

        # update target
        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(total_reward)

        if ep % max(1, episodes // 10) == 0:
            avg_last = np.mean(episode_rewards[-(episodes // 10):])
            print(f"Episode {ep}/{episodes}, avg reward (last chunk): {avg_last:.3f}")

    # after training, print learned policy (action with highest Q for each state)
    policy_net.eval()
    policy = {}
    for r in range(rows):
        for c in range(columns):
            s = (r, c)
            if s == blocked_cell:
                policy[s] = 'X'
                continue
            s_idx = state_to_index(s)
            v = one_hot_state(s_idx)
            with torch.no_grad():
                q = policy_net(torch.from_numpy(v).unsqueeze(0))
                a_idx = int(torch.argmax(q, dim=1).item())
                policy[s] = all_actions[a_idx]

    print('\nLearned policy: (rows x columns)')
    for r in range(rows):
        row_actions = [policy[(r, c)] for c in range(columns)]
        print(row_actions)

    return policy_net, episode_rewards


if __name__ == '__main__':
    # small smoke training to verify everything runs
    net, rewards = run_dqn(episodes=600, batch_size=32, buffer_capacity=2000)


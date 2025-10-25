# DQN Grid-World — README and Line-by-Line Guide

This README explains the `rl_2.py` file in detail (line-by-line and by logical blocks). The code implements a Deep Q-Network (DQN) for a small 3x4 grid-world that mirrors the original tabular Q-learning example. The goal of this document is to make every line and concept clear and actionable.

## Contents
- Quick summary
- Dependencies
- How to run
- High-level structure
- Line-by-line explanation of `rl_2.py`
- Frequently asked clarifications and tips

---

## Quick summary

`rl_2.py` replaces a tabular Q-learning solution with a DQN implementation using PyTorch. The environment is a deterministic grid with stochastic motion (intended move 0.8, left/right 0.1). The agent uses a small MLP to predict Q-values from a one-hot state encoding. Training uses an experience replay buffer and a target network updated periodically.

## Dependencies

- Python 3.8+ (tested on Python 3.x)
- NumPy
- PyTorch

Install the main dependency (PyTorch) with pip if needed:

```bash
pip install torch
```

(If you prefer a specific PyTorch build (CUDA) follow instructions at https://pytorch.org/.)

## How to run

From the repository root (where `RL_assign_2/rl_2.py` lives), run:

```bash
python3 RL_assign_2/rl_2.py
```

This performs a smoke training run for 600 episodes (default in the script) and prints progress and a final policy.

You can edit the call in `if __name__ == '__main__'` to change episodes, batch size, buffer size, etc.

---

## High-level structure of `rl_2.py`

1. Header docstring and imports
2. Environment parameters and transition model (grid, blocked cell, stochastic move outcomes)
3. Helper utilities (state indexing, one-hot vector creation)
4. DQN (PyTorch nn.Module) and Replay Buffer classes
5. Functions for action selection and single training step
6. `run_dqn()` — the training loop that ties everything together
7. `__main__` entrypoint that runs a small train session

---

## Line-by-line explanation

Below is a grouped, detailed explanation of each part of the code in `rl_2.py`. Lines are grouped by logical blocks. For clarity, I reproduce the exact code block and then provide an explanation for each statement inside that block.

### Header and imports

```python
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
```

- The triple-quoted string is a module-level docstring documenting purpose and install hint.
- `random` is Python's standard random module used for sampling and epsilon-greedy exploration.
- `deque` from `collections` is used to implement the replay buffer with a maximum length.
- `numpy` (imported as `np`) provides array utilities and numeric operations.
- `math` is used for math.exp when decaying epsilon.
- `torch`, `torch.nn`, and `torch.optim` are the core PyTorch modules for building and training the neural network.


### Environment parameters

```python
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
```

- `n_states = 12`: Number of distinct states in the grid (3x4 = 12 cells). The DQN input dimension uses this.
- `n_actions = 4`: There are 4 discrete actions: North, East, South, West.
- `goal_state = (0,3)`: A tuple (row, column) representing the terminal goal cell.
- `reward_goal = 1.0`: Reward received when reaching the goal.
- `penalty_state = (1,3)`: Another terminal cell (a negative terminal—the "penalty").
- `penalty = -1.0`: Reward when landing on the penalty cell.
- `step_cost = -0.04`: Small negative reward for each non-terminal step to encourage shorter paths.
- `rows, columns = 3, 4`: Grid shape (3 rows, 4 columns).
- `all_states = {...}`: Set comprehension that lists every (row, col) tuple.
- `all_actions = ["N","E","S","W"]`: Ordered list of action names; this order corresponds to action indices used throughout the code.
- `blocked_cell = (1,1)`: The grid cell that's blocked; moves into it keep the agent in place.
- `left_action` and `right_action`: Dictionaries mapping an intended action to the direction that would be a left or right slip; used to model stochastic transitions.


### Movement function and transition model

```python
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
```

- `move(state, action)`: Given a state tuple and an action string, compute the resulting state if that action occurred with no stochasticity. The function clamps movement at the grid borders (the agent cannot move off-grid). If the target cell is the blocked cell, the function returns the original state (move blocked).
- `transition_probs`: A dictionary mapping (state, action) -> list of (next_state, probability) pairs. For terminal states (goal or penalty), any action yields the same state with probability 1. Otherwise, the intended move occurs with probability 0.8, and the left and right slip moves occur with probability 0.1 each. This matches the classic stochastic grid-world model.

Notes: The code precomputes the transition distribution for all (state,action) pairs for fast sampling in the training loop.


### State index helper

```python
# helper maps
state_to_index = lambda s: s[0] * columns + s[1]
```

- `state_to_index`: A compact lambda that converts a (row, column) tuple into an integer index between 0 and 11 using row-major order. This index is used to create a one-hot vector for neural network input.


### DQN network class

```python
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
```

- `class DQN(nn.Module)`: A small multi-layer perceptron (MLP) implementing the Q-network. It inherits from PyTorch's `nn.Module`.
- `__init__(self, input_dim, output_dim, hidden=64)`: Constructor. `input_dim` is the dimension of the input (here `n_states`, the one-hot size). `output_dim` is number of actions (4). `hidden` controls the hidden layer width (default 64).
- `self.net = nn.Sequential(...)`: A sequential model with two hidden layers (hidden units each) and ReLU activations, and a final linear layer producing `output_dim` values (the predicted Q-values for each action).
- `forward(self, x)`: Applies the network to input `x` and returns the raw Q-value tensor. The network does not apply an activation to the final layer—Q-values are unconstrained real numbers.


### Replay buffer class

```python
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
```

- `ReplayBuffer` stores tuples of experience: `(state, action, reward, next_state, done)`.
- `deque(maxlen=capacity)` automatically forgets the oldest experiences when capacity is reached.
- `push(...)`: Appends one experience to the buffer.
- `sample(batch_size)`: Randomly samples `batch_size` experiences (without replacement) and returns NumPy arrays for each component arranged as batches.
- `__len__`: Allows `len(buffer)` to return the number of experiences currently stored.

Note: States and next_states are stored as NumPy one-hot vectors when they are pushed.


### Utility helpers: one-hot state, action selection

```python
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
```

- `one_hot_state`: Creates a float32 NumPy array with a 1.0 at the state's index and zeros elsewhere. This is the network input representation.
- `select_action`: Implements epsilon-greedy selection: with probability `epsilon` pick a random action index; otherwise, feed the state to the `policy_net` and return the action index with the highest Q-value. `torch.no_grad()` disables gradient tracking for inference.

Important detail: `policy_net(s)` expects a 2D tensor (batch dimension), so `unsqueeze(0)` adds that dimension.


### Single training step

```python
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
```

Step-by-step:
- If the buffer has fewer than `batch_size` experiences, we skip training and return `None`.
- `buffer.sample(batch_size)` returns NumPy arrays shaped (batch_size, input_dim) for states, (batch_size,) for actions, etc.
- The arrays are converted to PyTorch tensors and sent to `device` (CPU or GPU). Actions are converted to `long` for indexing; rewards and dones become float tensors.
- `q_values = policy_net(states_v).gather(1, actions_v.unsqueeze(1)).squeeze(1)`:
  - `policy_net(states_v)` outputs a tensor shape `(batch_size, n_actions)`.
  - `.gather(1, actions_v.unsqueeze(1))` selects the Q-value corresponding to each transition's taken action, yielding shape `(batch_size,1)`.
  - `.squeeze(1)` makes it shape `(batch_size,)`.
- `next_q_values = target_net(next_states_v).max(1)[0]` computes the maximum Q-value across actions from the target network for each next state.
- `target_q = rewards_v + gamma * next_q_values * (1.0 - dones_v)`: the Bellman target. If `done` is True (1.0), the bootstrap term is zeroed out.
- Mean squared error between predictions (`q_values`) and targets (`target_q`) is computed as the loss. The optimizer performs a gradient step.
- The function returns the scalar loss value for logging. If not enough data, returns `None`.

This is a standard DQN TD-update treating the target network as a fixed target for a step.


### Training loop: `run_dqn()`

```python
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
```

High-level explanation of `run_dqn`:

- Parameters allow hyperparameter customization (episodes, batch size, learning rate, replay buffer size, target update frequency, epsilon schedule, device).
- The function sets `device` to CUDA if available, else CPU.
- It constructs `policy_net` and `target_net` and copies the initial weights from `policy_net` to `target_net`.
- An Adam optimizer is created for `policy_net` parameters.
- It loops for `episodes` episodes. Each episode:
  - Starts at the fixed start state `(2,0)`.
  - Builds a one-hot vector for the state.
  - Runs a step loop with a max step cap (100) to avoid infinite episodes.
  - Epsilon is decayed via an exponential formula: `eps = eps_end + (eps_start - eps_end) * exp(-ep/eps_decay)`.
  - Selects an action using `select_action` (epsilon-greedy).
  - Samples the next state according to `transition_probs[(current_state, action)]` using `random.choices`.
  - Determines reward and `done` if next state is terminal (goal or penalty) or step cost otherwise.
  - Converts the next state to a one-hot vector and pushes the experience into the replay buffer.
  - Calls `train_step` to possibly update the `policy_net` using a batch from the replay buffer.
  - Updates `total_reward` and loops until `done` or the step cap is reached.
- Every `target_update` episodes the `target_net` weights are set to the `policy_net` weights (synchronization).
- The function prints progress once every `episodes // 10` episodes (10 updates across the run).
- After training completes, it runs the policy net for every state to pick the best action (highest predicted Q) and prints the learned policy array in row-major order.

Notes on the loop and implementation choices:
- The agent always starts at `(2, 0)` each episode.
- `random.choices` uses the precomputed transition probabilities to sample next states.
- Experiences store state vectors as NumPy one-hot arrays; sampling returns NumPy arrays and conversion to tensors happens inside `train_step`.
- Using a target network reduces oscillation and bootstrap bias.


### Entry point

```python
if __name__ == '__main__':
    # small smoke training to verify everything runs
    net, rewards = run_dqn(episodes=600, batch_size=32, buffer_capacity=2000)
```

- When the script is run directly, it calls `run_dqn` with 600 episodes, a batch size of 32, and a replay buffer cap of 2000 to perform a smoke training run and print results.

---

## Frequently asked clarifications and tips

1. Q: Why one-hot state vectors? A: The original grid-world has a small discrete state space (12 states), so one-hot encoding is a simple and direct way to feed the state into a neural network. For larger state representations (images or continuous features), you'd use different encoders.

2. Q: Why a target network and replay buffer? A: These are standard stability techniques in DQN: replay buffer breaks correlations between sequential samples; the target network provides a slowly-changing target for the temporal-difference update.

3. Q: Where to change the start state or rewards? A: Edit `current_state` initialization in `run_dqn`, or modify `reward_goal`, `penalty`, `step_cost` at the top of the file.

4. Q: How to speed up or debug training? A:
   - Reduce `episodes` to debug quickly (e.g., 50).
   - Set `buffer_capacity` to a smaller value for faster memory and less sampling diversity.
   - Lower `hidden` size in `DQN` constructor for faster forward/backward times.

5. Q: How to save or load trained models? A: After training you can call `torch.save(policy_net.state_dict(), "policy.pth")` and later load with `policy_net.load_state_dict(torch.load("policy.pth"))`.

6. Experiments to try:
   - Use Double DQN (compute argmax from policy_net and evaluate with target_net).
   - Use prioritized experience replay.
   - Change epsilon schedule to linear decay over steps instead of exponential by episode.

---

## Quick mapping to the original tabular Q-learning

- The tabular Q-learning stored Q-values in a NumPy array `Q[state_index, action]` and updated them with a TD rule.
- This DQN uses a function approximator (MLP) to estimate Q(s,a). The TD target formula is conceptually the same: r + gamma * max_a' Q_target(s', a') (with target network). Instead of updating a single Q-table entry, we compute gradients of the predicted Q-values and apply SGD to reduce MSE to the target.

---

## Final notes

If you want, I can:
- Add a `requirements.txt` listing `torch` and `numpy`.
- Add a small example script to load a trained model and visualize the policy.
- Add unit tests for transition probabilities and replay buffer sampling.

Tell me which additions you'd like and I'll update the repo.

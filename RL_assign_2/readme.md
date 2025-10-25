Title: DQN for 3×4 Stochastic Gridworld (with Adam + MSE)

Overview
- Goal: Learn an action-value function Q(s,a) in a 3×4 gridworld with stochastic transitions, a blocked cell, and two terminal states (+1 goal, −1 trap).
- Method: Deep Q-Network (DQN) with experience replay, target network, ε-greedy exploration, Adam optimizer, MSE loss, and one-hot state encoding over 12 states.
- Why DQN: Tabular Q-learning struggles to generalize; a neural network can share structure and is the standard baseline for function approximation in RL.

Environment and Notation

- Grid: rows = 3, columns = 4, total states n_states = 12.
- Terminals: goal_state = (0,3) with reward +1, penalty_state = (1,3) with reward −1.
- Stochastic action outcome: intended direction with 0.8 probability, left-slip 0.1, right-slip 0.1. Collisions with borders or blocked cell stay in place.
- Step cost: −0.04 for non-terminal moves.
- Action set: all_actions = ['N', 'E', 'S', 'W'].

File Structure

- Single Python script that:
  - Defines the gridworld transition model.
  - Implements DQN components (replay buffer, Q-network).
  - Trains the network and plots learning curves and a value heatmap.
  - Prints the greedy policy after training.

Section-by-Section Explanation

1) Imports
- numpy, random, matplotlib: environment math, RNG, plotting.
- torch, torch.nn, torch.optim: neural network building, loss, optimizer.
- collections.deque, namedtuple: replay buffer storage.

2) Environment constants
- n_states = 12: total states (3×4).
- n_actions = 4: N,E,S,W.
- goal_state, penalty_state, reward magnitudes, step_cost: define reward structure.
- rows, columns, all_states, all_actions: grid geometry and action space.
- blocked_cell = (1,1): impassable cell consistent with assignment diagram.

3) Slip model helpers
- left_action/right_action: maps intended action to its left/right neighbors on the compass for slip dynamics.

4) move(state, action)
- Computes the deterministic result of attempting to move; clamps to borders and enforces block logic by staying in place if moving into blocked cell.

5) transition_probs
- Precomputes P(s′|s,a) for all states and actions.
- For terminals: P((s→s), 1.0) so the agent stops.
- For non-terminals: intended 0.8, left 0.1, right 0.1 using move and the slip maps.

6) Utility functions
- state_to_index: converts 2D coordinate (r,c) into flat index r*columns + c; used to one-hot encode states and to index arrays.
- reward_of(state): returns +1, −1, or −0.04 depending on state.
- is_terminal(state): returns True at goal or trap.

7) ReplayBuffer
- Stores tuples (s, a, r, sp, done) with fixed max length.
- sample(batch_size): returns randomly sampled transitions to decorrelate updates and stabilize learning.

8) One-hot encoding
- one_hot(indices, depth): converts state indices to one-hot vectors of length 12; DQN input uses one-hot state features for simplicity.

9) QNet
- Small MLP: Linear(nS→64) → ReLU → Linear(64→64) → ReLU → Linear(64→nA).
- Outputs a vector of Q(s,·) for all actions given a one-hot state.

10) run_dqn(...) — Training loop
- Arguments (hyperparameters detailed below).
- Creates online network q and target network qt; copies q parameters into qt initially.
- Optimizer: Adam with the chosen learning rate.
- Loss: MSE between predicted Q(s,a) and target y.

Episode loop:
- Initialize s = (2,0) (start).
- ε-greedy action selection:
  - With probability ε pick a random action; else choose argmax_a Q(s,a).
- Environment step via transition_probs:
  - Sample s′ from P(·|s,a) using random.choices with given probabilities.
  - Compute reward r = reward_of(s′) and done flag d = is_terminal(s′).
- Store transition (s_idx, a_idx, r, sp_idx, d) into replay buffer.

Learning step (when buffer has enough data):
- Sample a minibatch of transitions.
- Build tensors:
  - S_oh, SP_oh: one-hot batches of s, s′.
  - A, R, D: action indices, rewards, done flags.
- Forward pass:
  - Qsa = q(S_oh).gather(1, A): current Q predictions for taken actions.
- Target computation (Double DQN target but MSE loss):
  - a_star = argmax q(S′) using online net (selection).
  - Qsp = qt(S′)[a_star] using target net (evaluation).
  - y = R + (1 − D) * γ * Qsp.
- Compute loss = MSE(Qsa, y), backpropagate, clip gradients to 1.0, optimizer step.
- Periodically hard-update target network qt ← q.

Bookkeeping:
- Track episodic return, training losses, and success indicator (reached goal).

11) Plotting and visualization
- Moving-average curves (window=50) for returns and success rate.
- Loss curve over training steps.
- Heatmap of V(s) = max_a Q(s,a) over the grid; NaN for the blocked cell; labels “T+1”/“T-1” for terminals.

12) Printing the greedy policy
- After training, for each non-terminal, non-blocked state, prints the action with maximum Q-value.

How the Data Flows Each Update

- Collect: Append (s,a,r,s′,done) to replay buffer as the agent interacts.
- Sample: Random minibatch breaks temporal correlations and improves stability.
- Compute target: y = r + γ max_a′ Q_target(s′,a′) if not terminal; y = r if terminal.
- Train: Minimize MSE between predicted Q(s,a) and y using Adam; periodically synchronize target network.

Hyperparameters and How to Tune Them

- episodes: Number of training episodes.
  - Higher means more data and usually better performance; typical: 1,000–10,000 for small tasks.
  - Symptom of too few: unstable policy, low success rate.

- max_steps: Max steps per episode.
  - Prevents endless wandering; 50–200 is typical here.
  - Too small: agent can’t reach the goal; too large: slower training.

- gamma (γ): Discount factor in [0,1).
  - Closer to 1 favors long-term rewards; 0.95–0.99 is common.
  - Too low: myopic behavior; too high: can slow learning or destabilize if targets become large.

- lr (learning rate): Adam step size.
  - 1e−3 is a strong default; try {1e−4, 5e−4, 1e−3}.
  - Too high: noisy or diverging loss; too low: slow learning.

- batch_size: Transitions per gradient step.
  - 32–128 typical; larger batches reduce variance but can smooth away useful signal in tiny tasks.

- start_learn: Number of transitions to collect before training starts.
  - Ensures initial minibatches are diverse; 500–2000 is typical for small buffers.

- target_update_every: Hard update period (in episodes here) for qt ← q.
  - Smaller values stabilize targets but can lag improvements; typical range 100–1000 environment steps. In this script it updates every fixed number of episodes; you can convert to steps if preferred.
  - Alternative: soft updates with τ (Polyak averaging): θ− ← τθ + (1−τ)θ−, τ≈0.005–0.02.

- eps_start, eps_end, eps_decay: ε-greedy exploration schedule.
  - eps_start: initial exploration (e.g., 1.0).
  - eps_end: floor on exploration (e.g., 0.05).
  - eps_decay: multiplicative decay per episode (e.g., 0.997).
  - Tuning: If the agent gets stuck exploiting too early, slow down decay (closer to 1). If training is slow, speed decay or raise eps_end slightly.

- seed: RNG seed for reproducibility.
  - Change to test robustness; keep fixed for deterministic comparisons.

- device: 'cpu' is fine for this small problem; 'cuda' for GPU if desired.

- Network width/depth (hidden=64):
  - Larger networks can overfit tiny problems; 32–128 per layer with 1–2 hidden layers is adequate here.
  - Indicators to adjust: persistent underfitting (raise capacity) or noisy/unstable learning (lower capacity and/or raise regularization like gradient clipping).

- Replay buffer capacity (inside ReplayBuffer):
  - 10k–50k for small tasks; too small can reintroduce correlation, too big may slow learning on non-stationary tasks (less relevant here).

- Loss function: MSELoss
  - MSE emphasizes large TD errors; can be sensitive to outliers.
  - If you observe occasional large spikes, try Huber (SmoothL1Loss) as a robust alternative without changing other components.

Common Pitfalls and Fixes

- Agent never reaches goal:
  - Increase episodes, slow ε decay, lower learning rate, or raise γ.
  - Verify transition model (slips, walls, block cell coordinates).

- Loss diverges or oscillates wildly:
  - Lower lr, add/strengthen gradient clipping, increase batch size slightly, or update target more often.

- Returns plateau at low value:
  - Adjust ε schedule (more exploration early), increase network capacity modestly, verify reward and terminal conditions, ensure one-hot state mapping is correct.

How to Run

- Ensure numpy, matplotlib, torch are installed.
- Run the script directly: python dqn_gridworld.py
- The script will:
  - Train for the configured number of episodes.
  - Show plots for episodic return, MSE loss, and success rate moving average.
  - Display a heatmap of V(s) and print the greedy policy.

Suggested Experiments for Your Report

- Compare MSE vs Huber: swap MSELoss with SmoothL1Loss and analyze stability.
- Vary γ ∈ {0.95, 0.99}, lr ∈ {1e−4, 5e−4, 1e−3}, batch_size ∈ {32, 64, 128}.
- Target update cadence: 100 vs 250 vs 500 episodes; or convert to step-based or soft updates with τ.
- Exploration schedule: test different eps_decay and eps_end values; plot success rates.
- Ablation: remove replay buffer (on-policy updates) to showcase why replay stabilizes training.

Reference Equations (for context)

- Target for non-terminal transitions:
  - y = r + γ max_{a′} Q_target(s′, a′)
- Loss (MSE on chosen actions in a batch):
  - L = mean[(Q_online(s,a) − y)^2]
- ε-greedy policy:
  - π(a|s) = ε/|A| + (1−ε)·1[a = argmax_a′ Q(s,a′)]


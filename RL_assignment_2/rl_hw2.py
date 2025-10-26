import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import time
import os

# ----------------------------
# Environment definitions
# ----------------------------
n_states = 12
n_actions = 4
goal_state = (0,3)
reward_goal = 1
penalty_state = (1,3)
penalty = -1
step_cost = -0.04

rows, columns = 3, 4
all_states = {(r,c) for r in range(rows) for c in range(columns)}
all_actions = ['N', 'E', 'S', 'W']
blocked_cell = (1,1)

left_action = {'N':'W', 'E':'N', 'S':'E', 'W':'S'}
right_action = {'N':'E', 'E':'S', 'S':'W', 'W':'N'}

def move(state, action):
    r, c = state
    if action == 'N':
        r2, c2 = max(r-1,0), c
    elif action == 'E':
        r2, c2 = r, min(c+1, columns-1)
    elif action == 'S':
        r2, c2 = min(r+1, rows-1), c    
    elif action == 'W':
        r2, c2 = r, max(c-1, 0)
    s2 = (r2, c2)
    if s2 == blocked_cell:
        s2 = state
    return s2

# Transition model (for stochastic next-state sampling)
transition_probs = {}
for state in all_states:
    for action in all_actions:
        if state == goal_state or state == penalty_state:
            transition_probs[(state, action)] = [(state, 1.0)]
            continue
        intended = move(state,action)
        left = move(state, left_action[action])
        right = move(state, right_action[action])
        transition_probs[(state, action)] = [(intended, 0.8), (left, 0.1), (right, 0.1)]

def state_to_index(s):
    return s[0] * columns + s[1]

def reward_of(state):
    if state == goal_state:
        return reward_goal
    if state == penalty_state:
        return penalty
    return step_cost

def is_terminal(state):
    return state == goal_state or state == penalty_state

# ----------------------------
# DQN components
# ----------------------------
Transition = namedtuple('Transition', ['s','a','r','sp','done'])

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buf = deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s  = np.array([b.s  for b in batch], dtype=np.int64)
        a  = np.array([b.a  for b in batch], dtype=np.int64)
        r  = np.array([b.r  for b in batch], dtype=np.float32)
        sp = np.array([b.sp for b in batch], dtype=np.int64)
        d  = np.array([b.done for b in batch], dtype=np.float32)
        return s,a,r,sp,d
    def __len__(self):
        return len(self.buf)

def one_hot(indices, depth):
    out = np.zeros((len(indices), depth), dtype=np.float32)
    out[np.arange(len(indices)), indices] = 1.0
    return out

class QNet(nn.Module):
    def __init__(self, nS, nA, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nS, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, nA)
        )
    def forward(self, x):
        return self.net(x)

# ----------------------------
# Training loop
# ----------------------------
def movavg(x, k=50):
    """Calculates the moving average of a 1D array."""
    if len(x) < k:
        k = len(x)
    if k == 0:
        return np.array([])
    # Convolve with a kernel of 1/k
    c = np.convolve(x, np.ones(k)/k, mode='valid')
    # Pad the beginning to match original length
    pad = np.concatenate([np.full(k-1, c[0]), c])
    return pad

def run_dqn(
    episodes=2500,
    max_steps=100,
    gamma=0.99,
    lr=1e-3,
    batch_size=64,
    start_learn=500,
    target_update_every=250,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.997,
    seed=123,
    device='cpu',
    hidden_size=64,
    save_plots=True  # Standardized argument to control saving plots
):
    """
    Runs the main DQN training loop.
    
    Args:
        ... (all hyperparameters) ...
        hidden_size (int): Number of neurons in hidden layers.
        save_plots (bool): If True, saves plots and Q-values. If False, runs quietly.
    
    Returns:
        (tuple): (moving_avg_returns, moving_avg_success_rate)
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    q = QNet(n_states, n_actions, hidden=hidden_size).to(device)
    qt = QNet(n_states, n_actions, hidden=hidden_size).to(device)
    qt.load_state_dict(q.state_dict())
    qt.eval()

    opt = optim.Adam(q.parameters(), lr=lr)
    mse = nn.MSELoss(reduction='mean')

    rb = ReplayBuffer(20000)

    eps = eps_start
    returns = []
    losses = []
    success = []

    for ep in range(episodes):
        s = (2,0)  # start
        G = 0.0
        reached_goal = 0

        for t in range(max_steps):
            si = state_to_index(s)
            if np.random.rand() < eps:
                a_idx = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    x = torch.from_numpy(one_hot([si], n_states)).to(device)
                    a_idx = int(torch.argmax(q(x), dim=1).item())

            # Sample next state using the stochastic transition model
            a_symbol = all_actions[a_idx]
            next_states, probs = zip(*transition_probs[(s, a_symbol)])
            sp = random.choices(next_states, probs)[0]
            r = reward_of(sp)
            d = float(is_terminal(sp))
            if sp == goal_state:
                reached_goal = 1

            rb.push(si, a_idx, r, state_to_index(sp), d)
            G += r
            s = sp

            # Minibatch Training
            if len(rb) >= start_learn:
                S, A, R, SP, D = rb.sample(batch_size)
                S_oh  = torch.from_numpy(one_hot(S,  n_states)).to(device)
                SP_oh = torch.from_numpy(one_hot(SP, n_states)).to(device)
                A_t   = torch.from_numpy(A).long().unsqueeze(1).to(device)
                R_t   = torch.from_numpy(R).unsqueeze(1).to(device)
                D_t   = torch.from_numpy(D).unsqueeze(1).to(device)

                # Q(s,a)
                Qsa = q(S_oh).gather(1, A_t)

                # Standard DQN Target
                with torch.no_grad():
                    Qsp = qt(SP_oh).max(dim=1, keepdim=True).values
                    Y = R_t + (1.0 - D_t) * gamma * Qsp

                loss = mse(Qsa, Y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 1.0)
                opt.step()

                losses.append(loss.item())

            if d == 1.0:
                break

        returns.append(G)
        success.append(reached_goal)

        # epsilon schedule
        eps = max(eps_end, eps * eps_decay)

        # target update
        if (ep + 1) % target_update_every == 0:
            qt.load_state_dict(q.state_dict())

    # Plots and Results 
    returns_ma = movavg(returns, 50)
    success_ma = movavg(success, 50)
    
    # Evaluation Results
    if save_plots:
        # Ensure 'plots' directory exists
        if not os.path.exists('plots'):
            os.makedirs('plots')
            
        fig, axs = plt.subplots(1,3, figsize=(16,4))
        
        # Plot 1: Episodic Return (Training Curve)
        axs[0].plot(returns, alpha=0.4, label='Return')
        axs[0].plot(returns_ma, label='MA(50)')
        axs[0].set_title('Episodic Return'); axs[0].legend()
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Cumulative Reward')

        # Plot 2: Training Loss
        loss_ma = movavg(losses, 200) # Use a larger window for loss
        if len(loss_ma) > 0:
            axs[1].plot(loss_ma, alpha=0.7)
        axs[1].set_title('Training Loss (MSE MA 200)')
        axs[1].set_xlabel('Training Step')
        axs[1].set_ylabel('Loss')

        # Plot 3: Success Rate (Evaluation Curve)
        axs[2].plot(success_ma)
        axs[2].set_ylim([0,1.05])
        axs[2].set_title('Success Rate MA(50)')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Success Rate')

        plt.tight_layout()
        plot_filename = 'plots/main_learning_curves.png'
        plt.savefig(plot_filename)
        print(f"Saved main learning curves to {plot_filename}")
        plt.close(fig) 

        # Heatmap of Q(s,a) for all actions
        with torch.no_grad():
            Q_vals = np.zeros((rows, columns, n_actions))
            for r in range(rows):
                for c in range(columns):
                    if (r,c) == blocked_cell:
                        Q_vals[r,c,:] = np.nan
                        continue
                    idx = state_to_index((r,c))
                    x = torch.from_numpy(one_hot([idx], n_states)).float().to(device)
                    Q_vals[r,c,:] = q(x).squeeze().cpu().numpy()

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('Learned Q(s,a) for each Action', fontsize=16)
        
        for i, action in enumerate(all_actions):
            ax = axs.flat[i]
            im = ax.imshow(Q_vals[:,:,i], cmap='viridis', origin='upper')
            plt.colorbar(im, ax=ax)
            ax.set_title(f'Action: {action}')
            for r in range(rows):
                for c in range(columns):
                    if (r,c) == blocked_cell: 
                        ax.text(c, r, 'WALL', ha='center', va='center', color='black', fontsize=10, weight='bold')
                        continue
                    if is_terminal((r,c)):
                         ax.text(c, r, 'TERM', ha='center', va='center', color='white', fontsize=10, weight='bold')
                    else:
                        txt = f'{Q_vals[r,c,i]:.2f}'
                        ax.text(c, r, txt, ha='center', va='center', color='w', fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = 'plots/q_function_heatmap.png'
        plt.savefig(plot_filename)
        print(f"Saved Q-function heatmap to {plot_filename}")
        plt.close(fig)

        # Print greedy policy and Q-values
        print("\n" + "="*30)
        print("Greedy policy & Q-values (N, E, S, W):")
        print("="*30)
        policy_grid = []
        q_value_details = []
        with torch.no_grad():
            for r in range(rows):
                row_txt_policy = []
                for c in range(columns):
                    s = (r,c)
                    if s == blocked_cell:
                        row_txt_policy.append('####')
                        q_value_details.append(f"({s[0]},{s[1]}) WALL")
                        continue
                    if is_terminal(s):
                        row_txt_policy.append('TERM')
                        q_value_details.append(f"({s[0]},{s[1]}) TERM")
                        continue
                    idx = state_to_index(s)
                    x = torch.from_numpy(one_hot([idx], n_states)).float().to(device)
                    q_values = q(x).squeeze()
                    a = torch.argmax(q_values).item()
                    row_txt_policy.append(all_actions[a])
                    
                    q_str = ", ".join([f"{q:.2f}" for q in q_values.cpu().numpy()])
                    q_value_details.append(f"({s[0]},{s[1]}) -> {all_actions[a]} [{q_str}]")
                policy_grid.append(" | ".join(row_txt_policy))
        
        for row_str in policy_grid:
            print(row_str)
        print("\nQ-Values Detail")
        for line in q_value_details:
            print(line)
        print("="*30 + "\n")

    return returns_ma, success_ma

def run_hyperparameter_comparison():
    """
    Runs experiments for different hyperparameters and saves comparison plots.
    """
    print("\nRunning Hyperparameter Comparison")
    # Ensure 'plots' directory exists
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    # Reduced episodes for faster comparison
    comparison_episodes = 1000 
    
    # 1. Compare Learning Rates
    lrs_to_test = [1e-2, 1e-3, 1e-4]
    lr_results = {}
    print(f"Testing Learning Rates (episodes={comparison_episodes})...")
    for lr in lrs_to_test:
        print(f"  Running with lr={lr}")
        start_time = time.time()
        returns_ma, _ = run_dqn(
            episodes=comparison_episodes, 
            lr=lr, 
            seed=42, 
            save_plots=False
        )
        lr_results[lr] = returns_ma
        print(f"  Finished in {time.time() - start_time:.2f}s")
        
    plt.figure(figsize=(10, 5))
    for lr, returns in lr_results.items():
        plt.plot(returns, label=f'lr={lr}')
    plt.title('Effect of Learning Rate on Episodic Return (MA 50)')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Cumulative Reward')
    plt.legend()
    plot_filename = 'plots/hyperparam_learning_rate.png'
    plt.savefig(plot_filename)
    print(f"Saved LR comparison plot to {plot_filename}")
    plt.close()

    # 2. Compare Discount Factors (gamma)
    gammas_to_test = [0.9, 0.95, 0.99]
    gamma_results = {}
    print(f"\nTesting Discount Factors (episodes={comparison_episodes})...")
    for gamma in gammas_to_test:
        print(f"  Running with gamma={gamma}")
        start_time = time.time()
        returns_ma, _ = run_dqn(
            episodes=comparison_episodes, 
            gamma=gamma, 
            seed=42, 
            save_plots=False
        )
        gamma_results[gamma] = returns_ma
        print(f"  Finished in {time.time() - start_time:.2f}s")
        
    plt.figure(figsize=(10, 5))
    for gamma, returns in gamma_results.items():
        plt.plot(returns, label=f'gamma={gamma}')
    plt.title('Effect of Discount Factor (gamma) on Episodic Return (MA 50)')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Cumulative Reward')
    plt.legend()
    plot_filename = 'plots/hyperparam_discount_factor.png'
    plt.savefig(plot_filename)
    print(f"Saved Gamma comparison plot to {plot_filename}")
    plt.close()

    # 3. Compare Batch Sizes
    batch_sizes_to_test = [16, 32, 64]
    batch_results = {}
    print(f"\nTesting Batch Sizes (episodes={comparison_episodes})...")
    for batch_size in batch_sizes_to_test:
        print(f"  Running with batch_size={batch_size}")
        start_time = time.time()
        returns_ma, _ = run_dqn(
            episodes=comparison_episodes, 
            batch_size=batch_size, 
            seed=42, 
            save_plots=False
        )
        batch_results[batch_size] = returns_ma
        print(f"  Finished in {time.time() - start_time:.2f}s")
        
    plt.figure(figsize=(10, 5))
    for batch_size, returns in batch_results.items():
        plt.plot(returns, label=f'batch_size={batch_size}')
    plt.title('Effect of Batch Size on Episodic Return (MA 50)')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Cumulative Reward')
    plt.legend()
    plot_filename = 'plots/hyperparam_batch_size.png'
    plt.savefig(plot_filename)
    print(f"Saved Batch Size comparison plot to {plot_filename}")
    plt.close()

    # 4. Compare Hidden Layer Sizes
    hidden_sizes_to_test = [32, 64] # Test 32 vs 64
    hidden_results = {}
    print(f"\nTesting Hidden Layer Sizes (episodes={comparison_episodes})...")
    for hidden_size in hidden_sizes_to_test:
        print(f"  Running with hidden_size={hidden_size}")
        start_time = time.time()
        returns_ma, _ = run_dqn(
            episodes=comparison_episodes, 
            hidden_size=hidden_size, 
            seed=42, 
            save_plots=False 
        )
        hidden_results[hidden_size] = returns_ma
        print(f"  Finished in {time.time() - start_time:.2f}s")
        
    plt.figure(figsize=(10, 5))
    for hidden_size, returns in hidden_results.items():
        plt.plot(returns, label=f'hidden_size={hidden_size}')
    plt.title('Effect of Hidden Layer Size on Episodic Return (MA 50)')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Cumulative Reward')
    plt.legend()
    plot_filename = 'plots/hyperparam_hidden_size.png'
    plt.savefig(plot_filename)
    print(f"Saved Hidden Size comparison plot to {plot_filename}")
    plt.close()
    
    print("Hyperparameter Comparison Finished")


if __name__ == '__main__':
    
    # Run the main, full experiment first
    print("Running Main Experiment")
    run_dqn(
        episodes=2500,
        max_steps=50,
        gamma=0.99,
        lr=1e-3,
        batch_size=32,
        start_learn=500,
        target_update_every=250,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.997,
        seed=123,
        device='cpu',
        hidden_size=64,
        save_plots=True 
    )
    
    # Run the hyperparameter comparison experiments
    run_hyperparameter_comparison()


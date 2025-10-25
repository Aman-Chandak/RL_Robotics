import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

# ----------------------------
# Your environment definitions
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
    device='cpu'
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    q = QNet(n_states, n_actions).to(device)
    qt = QNet(n_states, n_actions).to(device)
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

            # Learn
            if len(rb) >= start_learn:
                S, A, R, SP, D = rb.sample(batch_size)
                S_oh  = torch.from_numpy(one_hot(S,  n_states)).to(device)
                SP_oh = torch.from_numpy(one_hot(SP, n_states)).to(device)
                A_t   = torch.from_numpy(A).long().unsqueeze(1).to(device)
                R_t   = torch.from_numpy(R).unsqueeze(1).to(device)
                D_t   = torch.from_numpy(D).unsqueeze(1).to(device)

                # Q(s,a)
                Qsa = q(S_oh).gather(1, A_t)

                # Targets with Double DQN trick (still MSE to target)
                with torch.no_grad():
                    a_star = torch.argmax(q(SP_oh), dim=1, keepdim=True)
                    Qsp = qt(SP_oh).gather(1, a_star)
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

    # ----------- Plots -----------
    def movavg(x, k=50):
        if len(x) < k:
            return np.array(x)
        c = np.convolve(x, np.ones(k)/k, mode='valid')
        pad = np.concatenate([np.full(k-1, c[0]), c])
        return pad

    fig, axs = plt.subplots(1,3, figsize=(16,4))
    axs[0].plot(returns, alpha=0.4, label='Return')
    axs[0].plot(movavg(returns,50), label='MA(50)')
    axs[0].set_title('Episodic Return'); axs[0].legend()

    axs[1].plot(losses, alpha=0.7)
    axs[1].set_title('Training Loss (MSE)')

    sr = movavg(success, 50)
    axs[2].plot(sr)
    axs[2].set_ylim([0,1.05])
    axs[2].set_title('Success Rate MA(50)')

    plt.tight_layout()
    plt.show()

    # Heatmap of V(s)=max_a Q(s,a)
    with torch.no_grad():
        V = np.zeros((rows, columns))
        for r in range(rows):
            for c in range(columns):
                if (r,c) == blocked_cell:
                    V[r,c] = np.nan
                    continue
                idx = state_to_index((r,c))
                x = torch.from_numpy(one_hot([idx], n_states)).float()
                V[r,c] = q(x).max(dim=1).values.item()

    plt.figure(figsize=(5,3))
    im = plt.imshow(V, cmap='viridis', origin='upper')
    plt.colorbar(im)
    plt.title('Learned V(s) = max_a Q(s,a)')
    for r in range(rows):
        for c in range(columns):
            if (r,c)==blocked_cell: continue
            txt = 'T+1' if (r,c)==goal_state else ('T-1' if (r,c)==penalty_state else f'{V[r,c]:.2f}')
            plt.text(c, r, txt, ha='center', va='center', color='w', fontsize=10)
    plt.show()

    # Print greedy policy
    print("Greedy policy after training (N,E,S,W):")
    with torch.no_grad():
        for r in range(rows):
            row_txt = []
            for c in range(columns):
                s = (r,c)
                if s == blocked_cell:
                    row_txt.append('####')
                    continue
                if is_terminal(s):
                    row_txt.append('TERM')
                    continue
                idx = state_to_index(s)
                x = torch.from_numpy(one_hot([idx], n_states)).float()
                a = torch.argmax(q(x), dim=1).item()
                row_txt.append(all_actions[a])
            print(row_txt)

if __name__ == '__main__':
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
        device='cpu'
    )

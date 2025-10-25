import numpy as np
import random
import matplotlib.pyplot as plt

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

Q = np.zeros((n_states, n_actions)) #initialization

alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.2  # exploration rate
epochs = 10000


for epoch in range(epochs):
    current_state = (2,0)  # start state
    steps = 0
    while current_state != goal_state and current_state != penalty_state:
        steps += 1

        state_index = current_state[0] * columns + current_state[1]
        if np.random.rand() < epsilon:
            action = np.random.randint(0, n_actions)  # explore
        else:
            action = np.argmax(Q[state_index])  # exploit

        a = all_actions[action]
        next_states, probs = zip(*transition_probs[(current_state, a)])
        next_state = random.choices(next_states, probs)[0]
        

        # rewards
        if next_state == goal_state:
            reward = reward_goal
        elif next_state == penalty_state:
            reward = penalty
        else:
            reward = step_cost

        next_state_index = next_state[0] * columns + next_state[1]

       
        if next_state == goal_state or next_state == penalty_state:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state_index])

        Q[state_index, action] += alpha * (target - Q[state_index, action])

        
        if next_state == goal_state or next_state == penalty_state:
            break
        current_state = next_state

print("Learned Q-table:")
for r in range(rows):
    for c in range(columns):
        state_index = r * columns + c
        print(f"State ({r},{c}): ", end="")
        for a in range(n_actions):
            print(f"{all_actions[a]}: {Q[state_index, a]:.2f} ", end="")
        print() 
# for _ in range(Q.shape[0]):
#     best_action = np.argmax(Q[_])
#     r, c = divmod(_, columns)
#     print(f"State ({r},{c}): Best Action -> {all_actions[best_action]}")

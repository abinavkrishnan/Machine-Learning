# EECS 658 Assignment 8 - Parts 1, 2, 3, 4, 5, and 6
# Monte Carlo First Visit, Every Visit, On-Policy-Plus Every Visit, Q-Learning, SARSA,
# and Decaying Epsilon-Greedy Algorithms for Gridworld Task
# Author: Abinav Krishnan
# Student ID: 3068700
# Date: 12 / 7 / 2024

# Description: This program implements the RL Monte Carlo First Visit, Every Visit, 
# and On-Policy-Plus Every Visit algorithms, as well as Q-Learning, SARSA, and 
# Decaying Epsilon-Greedy algorithms to develop an optimal policy for the Gridworld task. 
# It prints the N(s), S(s), V(s), and Q(s, a) arrays at specified epochs/episodes and 
# shows the k, s, r, γ, and G(s) values. It also plots the error vs. t with ε labeled 
# for each algorithm.


import numpy as np
import matplotlib.pyplot as plt
import random

# Gridworld size and parameters
GRID_ROWS, GRID_COLS = 4, 6
TERMINAL_STATES = [(0, 5), (1, 5)]
GAMMA = 0.9
MAX_EPOCHS = 500
MAX_EPISODES = 500
EPSILON = 0.1
ALPHA = 0.1
DECAY_RATE = 0.99


# Actions: UP, DOWN, LEFT, RIGHT
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_EFFECTS = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

# Rewards function
def get_reward(state):
    if state in TERMINAL_STATES:
        return 0
    return -1

# Check if state is within bounds
def is_valid_state(state):
    return 0 <= state[0] < GRID_ROWS and 0 <= state[1] < GRID_COLS

# Generate an episode starting from a random state
def generate_episode(start_state):
    state = start_state
    episode = []
    while state not in TERMINAL_STATES:
        action = random.choice(ACTIONS)
        next_state = (state[0] + ACTION_EFFECTS[action][0], state[1] + ACTION_EFFECTS[action][1])
        if not is_valid_state(next_state):
            next_state = state  # Stay in the same state if hitting a wall
        reward = get_reward(next_state)
        episode.append((state, action, reward))
        state = next_state
    return episode

# Monte Carlo First Visit Algorithm
def monte_carlo_first_visit():
    print("\n--- Part 1: Monte Carlo First Visit Algorithm ---")
    N = np.zeros((GRID_ROWS, GRID_COLS))
    S = np.zeros((GRID_ROWS, GRID_COLS))
    V = np.zeros((GRID_ROWS, GRID_COLS))
    errors = []
    
    for epoch in range(MAX_EPOCHS):
        start_state = (random.randint(0, GRID_ROWS - 1), random.randint(0, GRID_COLS - 1))
        episode = generate_episode(start_state)
        G = 0
        visited_states = set()
        
        for t in range(len(episode) - 1, -1, -1):
            state, _, reward = episode[t]
            G = GAMMA * G + reward
            
            if state not in visited_states:
                visited_states.add(state)
                N[state] += 1
                S[state] += G
                V[state] = S[state] / N[state]
        
        if epoch % 10 == 0:
            error = np.max(np.abs(S / (N + 1e-10) - V))
            errors.append(error)
        
        if epoch in [0, 1, 10, MAX_EPOCHS - 1]:
            print(f"\nEpoch {epoch}")
            print("N(s):\n", N)
            print("S(s):\n", S)
            print("V(s):\n", V)
            print("\n" + "-"*85)
    
    plt.plot(range(0, MAX_EPOCHS, 10), errors)
    plt.xlabel('Epochs (t)')
    plt.ylabel('Error Value')
    plt.title('Error Value vs. Epochs (First Visit)')
    plt.grid(True)
    plt.show()

# Monte Carlo Every Visit Algorithm
def monte_carlo_every_visit():
    print("\n--- Part 2: Monte Carlo Every Visit Algorithm ---")
    N = np.zeros((GRID_ROWS, GRID_COLS))
    S = np.zeros((GRID_ROWS, GRID_COLS))
    V = np.zeros((GRID_ROWS, GRID_COLS))
    errors = []
    
    for epoch in range(MAX_EPOCHS):
        start_state = (random.randint(0, GRID_ROWS - 1), random.randint(0, GRID_COLS - 1))
        episode = generate_episode(start_state)
        G = 0
        
        for t in range(len(episode) - 1, -1, -1):
            state, _, reward = episode[t]
            G = GAMMA * G + reward
            
            N[state] += 1
            S[state] += G
            V[state] = S[state] / N[state]
        
        if epoch % 10 == 0:
            error = np.max(np.abs(S / (N + 1e-10) - V))
            errors.append(error)
        
        if epoch in [0, 1, 10, MAX_EPOCHS - 1]:
            print(f"\nEpoch {epoch}")
            print("N(s):\n", N)
            print("S(s):\n", S)
            print("V(s):\n", V)
            print("\n" + "-"*85)
    
    plt.plot(range(0, MAX_EPOCHS, 10), errors)
    plt.xlabel('Epochs (t)')
    plt.ylabel('Error Value')
    plt.title('Error Value vs. Epochs (Every Visit)')
    plt.grid(True)
    plt.show()

# Monte Carlo On-Policy-Plus Every Visit Algorithm
def monte_carlo_on_policy_plus():
    print("\n--- Part 3: Monte Carlo On-Policy-Plus Every Visit Algorithm ---")
    N = np.zeros((GRID_ROWS, GRID_COLS))
    S = np.zeros((GRID_ROWS, GRID_COLS))
    V = np.zeros((GRID_ROWS, GRID_COLS))
    policy = { (i, j): random.choice(ACTIONS) for i in range(GRID_ROWS) for j in range(GRID_COLS) if (i, j) not in TERMINAL_STATES }
    errors = []

    for epoch in range(MAX_EPOCHS):
        start_state = (random.randint(0, GRID_ROWS - 1), random.randint(0, GRID_COLS - 1))
        episode = generate_episode_with_policy(start_state, policy)
        G = 0

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = GAMMA * G + reward

            N[state] += 1
            S[state] += G
            V[state] = S[state] / N[state]

            # Update the policy to always choose the best action based on current estimates
            best_action = None
            best_value = float('-inf')
            for a in ACTIONS:
                next_state = (state[0] + ACTION_EFFECTS[a][0], state[1] + ACTION_EFFECTS[a][1])
                if is_valid_state(next_state):
                    if V[next_state] > best_value:
                        best_value = V[next_state]
                        best_action = a
            if best_action:
                policy[state] = best_action

        if epoch % 10 == 0:
            error = np.max(np.abs(S / (N + 1e-10) - V))
            errors.append(error)

        if epoch in [0, 1, 10, MAX_EPOCHS - 1]:
            print(f"\nEpoch {epoch}")
            print("N(s):\n", N)
            print("S(s):\n", S)
            print("V(s):\n", V)
            print("Policy:")
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    if (i, j) in policy:
                        print(f"State ({i}, {j}): {policy[(i, j)]}")
            print("\n" + "-" * 85)

    # Plot the error values
    plt.plot(range(0, MAX_EPOCHS, 10), errors)
    plt.xlabel('Epochs (t)')
    plt.ylabel('Error Value')
    plt.title('Error Value vs. Epochs (On-Policy-Plus Every Visit)')
    plt.grid(True)
    plt.show()

# Generate an episode based on the current policy with a step limit to avoid infinite loops
def generate_episode_with_policy(start_state, policy, max_steps=100):
    state = start_state
    episode = []
    steps = 0

    while state not in TERMINAL_STATES and steps < max_steps:
        action = policy[state]
        next_state = (state[0] + ACTION_EFFECTS[action][0], state[1] + ACTION_EFFECTS[action][1])
        if not is_valid_state(next_state):
            next_state = state  # Stay in the same state if hitting a wall
        reward = get_reward(next_state)
        episode.append((state, action, reward))
        state = next_state
        steps += 1

    return episode

# -----
# Rewards function for Q-Learning
def get_q_learning_reward(state):
    if state in TERMINAL_STATES:
        return 100
    return 0

# Q-Learning Algorithm
# Q-Learning Algorithm
def q_learning():
    print("\n--- Part 4: Q-Learning Algorithm ---")
    
    # Initialize Q-Table
    Q = { (i, j): np.zeros(len(ACTIONS)) for i in range(GRID_ROWS) for j in range(GRID_COLS) }
    errors = []
    rewards_matrix = np.zeros((GRID_ROWS, GRID_COLS))
    epsilon = 0.1  # Initialize epsilon as a local variable

    # Initialize Rewards Matrix (R)
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            if (i, j) in TERMINAL_STATES:
                rewards_matrix[i, j] = 100
            else:
                rewards_matrix[i, j] = 0

    print("Rewards Matrix (R):")
    print(rewards_matrix)

    for episode in range(MAX_EPISODES):
        state = (random.randint(0, GRID_ROWS - 1), random.randint(0, GRID_COLS - 1))
        total_error = 0

        while state not in TERMINAL_STATES:
            action = random.choice(ACTIONS) if random.uniform(0, 1) < epsilon else ACTIONS[np.argmax(Q[state])]
            next_state = (state[0] + ACTION_EFFECTS[action][0], state[1] + ACTION_EFFECTS[action][1])
            
            if not is_valid_state(next_state):
                next_state = state

            reward = get_q_learning_reward(next_state)
            best_next_action = np.max(Q[next_state])
            td_target = reward + GAMMA * best_next_action
            td_error = td_target - Q[state][ACTIONS.index(action)]
            Q[state][ACTIONS.index(action)] += ALPHA * td_error
            total_error += abs(td_error)

            state = next_state

        errors.append(total_error)
        epsilon *= DECAY_RATE  # Decay epsilon

        if episode in [0, 1, 10, MAX_EPISODES - 1]:
            print(f"\nEpisode {episode}")
            print("Q-Learning Value Matrix (Q):")
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    print(f"State ({i}, {j}): {Q[(i, j)]}")
            print("\n" + "-" * 85)

    plt.plot(range(MAX_EPISODES), errors)
    plt.xlabel('Episodes')
    plt.ylabel('Error Value')
    plt.title('Error Value vs. Episodes (Q-Learning)')
    plt.grid(True)
    plt.show()

# SARSA Algorithm
def sarsa():
    print("\n--- Part 5: SARSA Algorithm ---")
    
    # Initialize Q-Table
    Q = { (i, j): np.zeros(len(ACTIONS)) for i in range(GRID_ROWS) for j in range(GRID_COLS) }
    errors = []
    rewards_matrix = np.zeros((GRID_ROWS, GRID_COLS))
    epsilon = 0.1  # Initialize epsilon as a local variable

    # Initialize Rewards Matrix (R)
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            if (i, j) in TERMINAL_STATES:
                rewards_matrix[i, j] = 100
            else:
                rewards_matrix[i, j] = 0

    print("Rewards Matrix (R):")
    print(rewards_matrix)

    for episode in range(MAX_EPISODES):
        state = (random.randint(0, GRID_ROWS - 1), random.randint(0, GRID_COLS - 1))
        action = random.choice(ACTIONS) if random.uniform(0, 1) < epsilon else ACTIONS[np.argmax(Q[state])]
        total_error = 0

        while state not in TERMINAL_STATES:
            next_state = (state[0] + ACTION_EFFECTS[action][0], state[1] + ACTION_EFFECTS[action][1])

            if not is_valid_state(next_state):
                next_state = state

            reward = get_q_learning_reward(next_state)
            next_action = random.choice(ACTIONS) if random.uniform(0, 1) < epsilon else ACTIONS[np.argmax(Q[next_state])]
            
            td_target = reward + GAMMA * Q[next_state][ACTIONS.index(next_action)]
            td_error = td_target - Q[state][ACTIONS.index(action)]
            Q[state][ACTIONS.index(action)] += ALPHA * td_error
            total_error += abs(td_error)

            state, action = next_state, next_action

        errors.append(total_error)
        epsilon *= DECAY_RATE  # Decay epsilon

        # Print the Q matrix at episode 0, 1, 10, and the final episode
        if episode in [0, 1, 10, MAX_EPISODES - 1]:
            print(f"\nEpisode {episode}")
            print("SARSA Value Matrix (Q):")
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    print(f"State ({i}, {j}): {Q[(i, j)]}")
            print("\n" + "-" * 85)

    # Plot the error values
    plt.plot(range(MAX_EPISODES), errors)
    plt.xlabel('Episodes')
    plt.ylabel('Error Value')
    plt.title('Error Value vs. Episodes (SARSA)')
    plt.grid(True)
    plt.show()

# Initialize Q-Table
def initialize_q_table():
    return { (i, j): np.zeros(len(ACTIONS)) for i in range(GRID_ROWS) for j in range(GRID_COLS) }

# Decaying Epsilon-Greedy Algorithm
def decaying_epsilon_greedy():
    print("\n--- Part 6: Decaying Epsilon-Greedy Algorithm ---")
    
    # Initialize Q-Table
    Q = initialize_q_table()
    errors = []
    rewards_matrix = np.zeros((GRID_ROWS, GRID_COLS))
    epsilon = 1.0  # Start with epsilon = 1.0 for maximum exploration

    # Initialize Rewards Matrix (R)
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            if (i, j) in TERMINAL_STATES:
                rewards_matrix[i, j] = 100
            else:
                rewards_matrix[i, j] = 0

    print("Rewards Matrix (R):")
    print(rewards_matrix)

    for episode in range(MAX_EPISODES):
        state = (random.randint(0, GRID_ROWS - 1), random.randint(0, GRID_COLS - 1))
        total_error = 0

        while state not in TERMINAL_STATES:
            # Epsilon-greedy action selection
            action = random.choice(ACTIONS) if random.uniform(0, 1) < epsilon else ACTIONS[np.argmax(Q[state])]
            next_state = (state[0] + ACTION_EFFECTS[action][0], state[1] + ACTION_EFFECTS[action][1])

            if not is_valid_state(next_state):
                next_state = state  # Stay in the same state if hitting a wall

            reward = get_reward(next_state)
            best_next_action = np.max(Q[next_state])
            td_target = reward + GAMMA * best_next_action
            td_error = td_target - Q[state][ACTIONS.index(action)]
            Q[state][ACTIONS.index(action)] += ALPHA * td_error
            total_error += abs(td_error)

            state = next_state

        errors.append(total_error)
        epsilon *= DECAY_RATE  # Decay epsilon after each episode

        # Print the Q matrix at episode 0, 1, 10, and the final episode
        if episode in [0, 1, 10, MAX_EPISODES - 1]:
            print(f"\nEpisode {episode}")
            print("Decaying Epsilon-Greedy Value Matrix (Q):")
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    print(f"State ({i}, {j}): {Q[(i, j)]}")
            print("\n" + "-" * 85)

    # Plot the error values
    plt.plot(range(MAX_EPISODES), errors)
    plt.xlabel('Episodes')
    plt.ylabel('Error Value')
    plt.title('Error Value vs. Episodes (Decaying Epsilon-Greedy)')
    plt.grid(True)
    plt.show()

# Run both algorithms
if __name__ == "__main__":
    monte_carlo_first_visit()
    monte_carlo_every_visit()
    monte_carlo_on_policy_plus()
    q_learning()
    sarsa()
    decaying_epsilon_greedy()
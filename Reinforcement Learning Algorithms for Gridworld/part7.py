# EECS 658 Assignment 8 - Part 7
# Cumulative Average Reward Comparison for Q-Learning, SARSA, and Decaying Epsilon-Greedy
# Author: Abinav Krishnan
# Student ID: 3068700
# Date: 12 / 7 / 2024
# Description: This program compares the cumulative average rewards of Q-Learning, SARSA, 
# and Decaying Epsilon-Greedy algorithms over a Gridworld task.

import numpy as np
import matplotlib.pyplot as plt
import random

# Gridworld size and parameters
GRID_ROWS, GRID_COLS = 4, 6
TERMINAL_STATES = [(0, 5), (1, 5)]
GAMMA = 0.9
MAX_EPISODES = 500
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
        return 100
    return 0

# Check if state is within bounds
def is_valid_state(state):
    return 0 <= state[0] < GRID_ROWS and 0 <= state[1] < GRID_COLS

# Initialize Q-Table
def initialize_q_table():
    return { (i, j): np.zeros(len(ACTIONS)) for i in range(GRID_ROWS) for j in range(GRID_COLS) }

# Q-Learning Algorithm
def q_learning():
    Q = initialize_q_table()
    epsilon = 0.1
    rewards = []

    for episode in range(MAX_EPISODES):
        state = (random.randint(0, GRID_ROWS - 1), random.randint(0, GRID_COLS - 1))
        total_reward = 0

        while state not in TERMINAL_STATES:
            action = random.choice(ACTIONS) if random.uniform(0, 1) < epsilon else ACTIONS[np.argmax(Q[state])]
            next_state = (state[0] + ACTION_EFFECTS[action][0], state[1] + ACTION_EFFECTS[action][1])
            if not is_valid_state(next_state):
                next_state = state

            reward = get_reward(next_state)
            total_reward += reward

            best_next_action = np.max(Q[next_state])
            Q[state][ACTIONS.index(action)] += ALPHA * (reward + GAMMA * best_next_action - Q[state][ACTIONS.index(action)])
            state = next_state

        rewards.append(total_reward)
        epsilon *= DECAY_RATE

    return rewards

# SARSA Algorithm
def sarsa():
    Q = initialize_q_table()
    epsilon = 0.1
    rewards = []

    for episode in range(MAX_EPISODES):
        state = (random.randint(0, GRID_ROWS - 1), random.randint(0, GRID_COLS - 1))
        action = random.choice(ACTIONS) if random.uniform(0, 1) < epsilon else ACTIONS[np.argmax(Q[state])]
        total_reward = 0

        while state not in TERMINAL_STATES:
            next_state = (state[0] + ACTION_EFFECTS[action][0], state[1] + ACTION_EFFECTS[action][1])
            if not is_valid_state(next_state):
                next_state = state

            reward = get_reward(next_state)
            total_reward += reward

            next_action = random.choice(ACTIONS) if random.uniform(0, 1) < epsilon else ACTIONS[np.argmax(Q[next_state])]
            Q[state][ACTIONS.index(action)] += ALPHA * (reward + GAMMA * Q[next_state][ACTIONS.index(next_action)] - Q[state][ACTIONS.index(action)])
            state, action = next_state, next_action

        rewards.append(total_reward)
        epsilon *= DECAY_RATE

    return rewards

# Decaying Epsilon-Greedy Algorithm
def decaying_epsilon_greedy():
    Q = initialize_q_table()
    epsilon = 1.0
    rewards = []

    for episode in range(MAX_EPISODES):
        state = (random.randint(0, GRID_ROWS - 1), random.randint(0, GRID_COLS - 1))
        total_reward = 0

        while state not in TERMINAL_STATES:
            action = random.choice(ACTIONS) if random.uniform(0, 1) < epsilon else ACTIONS[np.argmax(Q[state])]
            next_state = (state[0] + ACTION_EFFECTS[action][0], state[1] + ACTION_EFFECTS[action][1])
            if not is_valid_state(next_state):
                next_state = state

            reward = get_reward(next_state)
            total_reward += reward

            best_next_action = np.max(Q[next_state])
            Q[state][ACTIONS.index(action)] += ALPHA * (reward + GAMMA * best_next_action - Q[state][ACTIONS.index(action)])
            state = next_state

        rewards.append(total_reward)
        epsilon *= DECAY_RATE

    return rewards

# Plot cumulative average rewards for Q-Learning, SARSA, and Decaying Epsilon-Greedy
def plot_cumulative_average_rewards(q_rewards, sarsa_rewards, decaying_rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(q_rewards) / (np.arange(len(q_rewards)) + 1), label='Q-Learning')
    plt.plot(np.cumsum(sarsa_rewards) / (np.arange(len(sarsa_rewards)) + 1), label='SARSA')
    plt.plot(np.cumsum(decaying_rewards) / (np.arange(len(decaying_rewards)) + 1), label='Decaying Epsilon-Greedy')

    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Average Reward')
    plt.title('Cumulative Average Reward Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run all algorithms and plot the cumulative reward comparison
if __name__ == "__main__":
    q_rewards = q_learning()
    sarsa_rewards = sarsa()
    decaying_rewards = decaying_epsilon_greedy()
    plot_cumulative_average_rewards(q_rewards, sarsa_rewards, decaying_rewards)

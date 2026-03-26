# EECS 658 Assignment 7
# Name: Abinav Krishnan
# Student ID: 3068700
# Date: 11/21/2024
# Description: This assignment is about policy iteration.
# Inputs: None | Outputs: Matricies, Graphs

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

# Constants
GRID_SIZE = 5
DISCOUNT_FACTOR = 1
REWARD = -1
TERMINAL_STATE_REWARD = 0
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_PROBABILITY = 0.25  # Equal probability for all actions
TERMINAL_STATES = [(0, 0), (4, 4)]

# Helper functions
def is_terminal_state(state):
    return state in TERMINAL_STATES

def get_next_state(state, action):
    x, y = state
    if action == 'up':
        return (max(0, x - 1), y)
    elif action == 'down':
        return (min(GRID_SIZE - 1, x + 1), y)
    elif action == 'left':
        return (x, max(0, y - 1))
    elif action == 'right':
        return (x, min(GRID_SIZE - 1, y + 1))
    return state

# Initialize values and policy
values = np.zeros((GRID_SIZE, GRID_SIZE))
policy = np.full((GRID_SIZE, GRID_SIZE), 'up', dtype=object)

# Policy Iteration Algorithm with Error Tracking
def policy_iteration():
    global values, policy
    stable_policy = False
    iteration = 0
    error_list = []

    while not stable_policy:
        print(f"Iteration {iteration}:")
        print("Values:")
        print(values)
        print("Policy:")
        print(policy)
        stable_policy = True
        max_error = 0

        # Policy Evaluation
        for _ in range(100):  # Arbitrary large number for convergence
            updated_values = np.copy(values)
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    state = (x, y)
                    if is_terminal_state(state):
                        updated_values[x, y] = TERMINAL_STATE_REWARD
                        continue
                    
                    # Calculate value using stochastic policy
                    value = 0
                    for action in ACTIONS:
                        next_state = get_next_state(state, action)
                        reward = TERMINAL_STATE_REWARD if is_terminal_state(next_state) else REWARD
                        value += ACTION_PROBABILITY * (reward + DISCOUNT_FACTOR * values[next_state])
                    updated_values[x, y] = value
            
            max_error = max(max_error, np.max(np.abs(updated_values - values)))
            values = updated_values

            # Track the maximum error per iteration
            error_list.append(max_error)

            # Break if the change is small enough
            if max_error < 1e-3:
                break

        # Policy Improvement
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                state = (x, y)
                if is_terminal_state(state):
                    continue
                
                best_action = None
                best_value = float('-inf')
                for action in ACTIONS:
                    next_state = get_next_state(state, action)
                    reward = TERMINAL_STATE_REWARD if is_terminal_state(next_state) else REWARD
                    value = reward + DISCOUNT_FACTOR * values[next_state]
                    if value > best_value:
                        best_value = value
                        best_action = action
                
                # Update the policy if it changes
                if best_action != policy[x, y]:
                    stable_policy = False
                policy[x, y] = best_action

        iteration += 1

    # Plot the Error Values
    plt.figure(figsize=(10, 6))
    plt.plot(error_list, label="Error per Iteration")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title("Error Values During Policy Iteration")
    plt.legend()
    plt.grid()
    plt.show()

# Run Policy Iteration
policy_iteration()

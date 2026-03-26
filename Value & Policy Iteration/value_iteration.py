# EECS 658 Assignment 7
# Name: Abinav Krishnan
# Student ID: 3068700
# Date: 11/21/2024
# Description: This assignment demonstrates the implementation of the Value Iteration algorithm for a grid-world problem.
# Inputs: None | Outputs: Matrices (values and policy), Graph (error over iterations)

import numpy as np
import matplotlib.pyplot as plt

# Constants defining the grid-world environment
GRID_SIZE = 5  # Size of the grid world (5x5)
DISCOUNT_FACTOR = 1  # Discount factor (gamma) to weight future rewards
REWARD = -1  # Reward for transitioning to any non-terminal state
TERMINAL_STATE_REWARD = 0  # Fixed reward for terminal states
ACTIONS = ['up', 'down', 'left', 'right']  # Available actions for movement

# Define terminal states
TERMINAL_STATES = [(0, 0), (4, 4)]  # (row, column) positions of terminal states

# Helper function to check if a state is a terminal state
def is_terminal_state(state):
    return state in TERMINAL_STATES

# Helper function to determine the next state based on the current state and action
def get_next_state(state, action):
    x, y = state
    if action == 'up':  # Move up if within bounds
        return (max(0, x - 1), y)
    elif action == 'down':  # Move down if within bounds
        return (min(GRID_SIZE - 1, x + 1), y)
    elif action == 'left':  # Move left if within bounds
        return (x, max(0, y - 1))
    elif action == 'right':  # Move right if within bounds
        return (x, min(GRID_SIZE - 1, y + 1))
    return state  # Default case (no movement)

# Initialize the value function for all states
values = np.zeros((GRID_SIZE, GRID_SIZE))  # All states start with a value of zero

# Value Iteration Algorithm
def value_iteration():
    """
    Perform value iteration to compute optimal state values.
    Returns:
        A list of errors (maximum change in values) per iteration for convergence analysis.
    """
    global values
    iteration = 0  # Track the number of iterations
    error_list = []  # Store the maximum value change (error) per iteration

    while True:
        updated_values = np.copy(values)  # Temporary copy to update the value function
        max_error = 0  # Track the largest change in state values for this iteration

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                state = (x, y)
                if is_terminal_state(state):
                    # Terminal states have fixed values (TERMINAL_STATE_REWARD)
                    updated_values[x, y] = TERMINAL_STATE_REWARD
                    continue

                # Apply the Bellman update equation
                best_value = float('-inf')  # Initialize with the smallest possible value
                for action in ACTIONS:
                    next_state = get_next_state(state, action)  # Determine the next state
                    reward = REWARD  # Fixed reward for non-terminal states
                    value = reward + DISCOUNT_FACTOR * values[next_state]  # Compute value
                    best_value = max(best_value, value)  # Track the maximum value

                # Update the value of the current state
                updated_values[x, y] = best_value
                # Track the maximum change (error) in value updates
                max_error = max(max_error, abs(updated_values[x, y] - values[x, y]))

        # Update the value function after processing all states
        values[:] = updated_values
        error_list.append(max_error)
        iteration += 1

        # Debug: Print values grid at each iteration
        print(f"Iteration {iteration}:")
        print(values)

        # Stop iterating if the maximum value change is below the convergence threshold
        if max_error < 1e-3:
            break

    return error_list

# Run the Value Iteration algorithm
error_values = value_iteration()

# Display the final values after convergence
print("Final Values (Converged):")
for row in values:
    print(" ".join(f"{int(cell)}" for cell in row))  # Display as integers for simplicity

# Plot the error values to show convergence
plt.figure()
plt.plot(range(len(error_values)), error_values, marker='o', label='Error Values')  # Plot error over iterations
plt.title('Error Values vs Iterations')  # Title of the plot
plt.xlabel('Iteration')  # X-axis label
plt.ylabel('Error Value')  # Y-axis label
plt.grid()  # Add grid lines for better readability
plt.legend()  # Add legend to the plot
plt.show()

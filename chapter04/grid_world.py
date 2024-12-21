#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# ITERATIVE POLICY EVALUATION

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use('Agg')

WORLD_SIZE = 4
# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROB = 0.25


def is_terminal(state):
    x, y = state
    return (x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)


def step(state, action):
    # check if state is terminal, in which case action is ineffective
    if is_terminal(state):
        return state, 0

    # Apply action
    next_state = (np.array(state) + action).tolist()
    x, y = next_state

    # check if out of bounds, in which case action is undone
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        next_state = state

    reward = -1
    return next_state, reward


def draw_image(image):
    _, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

        # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')
    ax.add_table(tb)


def compute_state_value(in_place=True, discount=1.0, n_iterations=None):
    '''
    Computes V(s) by Iterative Policy Evaluation
    '''
    # Initialize V(s) to zero
    new_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    iteration = 0
    while True:
        if in_place:
            # let V(s) be updated while iterating over states
            state_values = new_state_values
        else:
            # update V(s) only at the end of a complete iteration
            state_values = new_state_values.copy()
        
        # keep copy of current V(s) so that we can later check convergence criterion
        old_state_values = state_values.copy()

        # Iterate over all states
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                # Initialize current valuation
                value = 0
                # Iterate over all actions
                for action in ACTIONS:
                    # Get s', r
                    (next_i, next_j), reward = step([i, j], action)
                    # Add r + gamma*V(s') to valuation
                    value += ACTION_PROB * (reward + discount * state_values[next_i, next_j])
                # Update V(s)
                new_state_values[i, j] = value

        # Update number of iterations
        iteration += 1

        # Compute max update difference
        max_delta_value = abs(old_state_values - new_state_values).max()

        # Terminate if either update diff is tiny or we reached the max allowed number of iterations
        if max_delta_value < 1e-4 or iteration==n_iterations:
            break

    # Report results
    return new_state_values, iteration


def figure_4_1(n_iterations=np.Infinity):
    # While the author suggests using in-place iterative policy evaluation,
    # Figure 4.1 actually uses out-of-place version.
    _, asycn_iteration = compute_state_value(in_place=True, n_iterations=n_iterations)
    values, sync_iteration = compute_state_value(in_place=False, n_iterations=n_iterations)
    draw_image(np.round(values, decimals=1))
    print('In-place: {} iterations'.format(asycn_iteration))
    print('Synchronous: {} iterations'.format(sync_iteration))

    plt.savefig('../images/figure_4_1.png')
    plt.close()


if __name__ == '__main__':
    figure_4_1()



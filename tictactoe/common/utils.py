import numpy as np
from termcolor import colored


def check_if_done(state):
    # winning states
    if state[0] == state[1] == state[2] != 0:
        return True, state[0]
    if state[3] == state[4] == state[5] != 0:
        return True, state[3]
    if state[6] == state[7] == state[8] != 0:
        return True, state[6]
    if state[0] == state[3] == state[6] != 0:
        return True, state[0]
    if state[1] == state[4] == state[7] != 0:
        return True, state[1]
    if state[2] == state[5] == state[8] != 0:
        return True, state[2]
    if state[0] == state[4] == state[8] != 0:
        return True, state[0]
    if state[2] == state[4] == state[6] != 0:
        return True, state[2]

    # tie
    if state.all():
        return True, 0

    # game still on...
    return False, None


def print_board(state):
    map_text = {1: ' X ', -1: ' O ', 0: '   '}
    for row in range(3):
        print(colored('-------------\n|', 'white', attrs=['bold']), end='')
        for col in range(3):
            idx = row * 3 + col
            print(colored(map_text[state[idx]], 'white', attrs=['bold']), end='')
            print(colored('|', 'white', attrs=['bold']), end='')
        print()
    print(colored('-------------', 'white', attrs=['bold']))
    print()


def print_board_with_action_values(state, action_values):
    action_values_normalized = action_values / np.linalg.norm(action_values)

    map_text = {1: ' X ', -1: ' O ', 0: '   '}
    for row in range(3):
        print(colored('-------------\n|', 'white', attrs=['bold']), end='')
        for col in range(3):
            idx = row * 3 + col
            if action_values_normalized[idx] <= 0.33:
                background_color = 'red'
            elif action_values_normalized[idx] >= 0.67:
                background_color = 'green'
            else:
                background_color = 'yellow'
            print(colored(map_text[state[idx]], background_color, attrs=['reverse', 'bold']), end='')
            print(colored('|', 'white', attrs=['bold']), end='')
        print()
    print(colored('-------------', 'white', attrs=['bold']))
    print()

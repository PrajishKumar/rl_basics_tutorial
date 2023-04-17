import random

import numpy as np

from utils import check_if_done, print_board


class GameEnv:
    def __init__(self):
        self.state = np.zeros(9)
        self.actions = []

    def reset(self):
        self.state = np.zeros(9)

        # flip a coin to decide who starts first.
        if random.random() > 0.5:
            self.state[random.randint(0, 8)] = -1

    def step(self, action: int):
        """
        :return: <state>, <reward>, <done>, <next_state>
        """

        self.actions.append(action)

        # terminate with a big penalty if we try to play in a used spot.
        if self.state[action] != 0:
            return self.state, -10., True, self.state

        prev_state = np.copy(self.state)

        # check if we already won with our action.
        self.state[action] = 1
        game_over, winner = check_if_done(self.state)
        if game_over:
            return prev_state, 100. * float(winner), True, self.state

        # otherwise, let opponent play
        free_spots = np.where(self.state == 0)[0]
        chosen_spot = np.random.choice(free_spots)
        self.state[chosen_spot] = -1

        # check if opponent's play concludes the game.
        game_over, winner = check_if_done(self.state)
        if game_over:
            return prev_state, 100. * float(winner), True, self.state

        # otherwise, game continues.
        return prev_state, 0., False, self.state

    def get_state(self):
        return self.state

    def print_board(self):
        print_board(self.state)

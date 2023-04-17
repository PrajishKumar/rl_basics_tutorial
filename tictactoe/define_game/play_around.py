import numpy as np
import random

from tictactoe.common.utils import check_if_done, print_board


class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9)

    def agent_play(self, spot: int):
        if self.board[spot] == 0:
            self.board[spot] = 1

    def opponent_play(self):
        chosen_spot_ = np.random.choice(self.get_empty_spots())
        self.board[chosen_spot_] = -1

    def get_empty_spots(self):
        return np.where(self.board == 0)[0]


if __name__ == '__main__':
    game = TicTacToe()

    agent_is_smart = False

    done = False
    winner = None
    while not done:
        if agent_is_smart:
            # Choose among the empty spots.
            chosen_spot = np.random.choice(game.get_empty_spots())
            game.agent_play(chosen_spot)
        else:
            # Choose any random spot.
            game.agent_play(random.randint(0, 8))

        done, winner = check_if_done(game.board)
        if done:
            break

        game.opponent_play()

        done, winner = check_if_done(game.board)

    print_board(game.board)
    if winner == 1:
        print("YOU WIN!")
    elif winner == -1:
        print("YOU LOST :(")
    else:
        print("TIE!")

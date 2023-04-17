import numpy as np
import random

from env import GameEnv


def get_action(state, agent_is_smart=False):
    if agent_is_smart:
        # choose among the empty spots.
        return np.random.choice(np.where(state == 0)[0])
    else:
        # choose any random spot.
        return random.randint(0, 8)


def try_once():
    env = GameEnv()
    env.reset()

    done = False
    state = env.get_state()

    print("START")
    env.print_board()

    while not done:
        # choose an action
        action = get_action(state=state)

        # step into environment
        state, reward, done, next_state = env.step(action=action)

        # debug prints
        print(f"Agent chose position: {action}")
        env.print_board()
        print(f"Reward received: {reward}\n")

        state = next_state


def try_many_times(num=1000):
    env = GameEnv()
    num_games_total = 0
    num_games_won = 0
    num_games_lost_smartly = 0
    num_games_lost_stupidly = 0
    num_games_tied = 0

    for _ in range(num):
        env.reset()
        done = False
        reward = 0.
        state = env.get_state()
        while not done:
            # choose an action
            action = get_action(state=state)

            # step into environment
            state, reward, done, next_state = env.step(action=action)

            state = next_state

        if reward == 100.:
            num_games_won += 1
        elif reward == -100.:
            num_games_lost_smartly += 1
        elif reward == -10.:
            num_games_lost_stupidly += 1
        else:
            num_games_tied += 1
        num_games_total += 1

    print(f"Total games                    : {num_games_total}")
    print(f"Won                            : {num_games_won}")
    print(f"Lost, but fought like a warrior: {num_games_lost_smartly}")
    print(f"Lost, but had a brain of a pea : {num_games_lost_stupidly}")
    print(f"Tied                           : {num_games_tied}")


if __name__ == '__main__':
    try_many_times()

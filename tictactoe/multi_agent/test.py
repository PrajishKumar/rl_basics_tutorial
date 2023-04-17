import numpy as np
import pathlib
import torch

from utils import print_board, print_board_with_action_values

from env import GameEnv
from agent import Agent

env = GameEnv()
env.reset()

agent = Agent()
agent.actor.load_state_dict(torch.load(f'{pathlib.Path(__file__).parent.resolve()}/actor.pt'))
agent.critic.load_state_dict(torch.load(f'{pathlib.Path(__file__).parent.resolve()}/critic.pt'))


def get_action(state, agent):
    return agent.act(state, eps=0.)


def main():
    try_many_times()


def try_once():
    done = False
    state = env.get_state()

    print("START")
    env.print_board()

    while not done:
        # choose an action
        action = get_action(state=state, agent=agent)

        # step into environment
        state, reward, done, next_state = env.step(action=action)

        # debug prints
        print(f"Agent chose position: {action}")
        env.print_board()
        print(f"Reward received: {reward}\n")

        state = next_state


def try_many_times(num=1000):
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
            action = get_action(state=state, agent=agent)

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
    main()

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import torch

from collections import deque

from env import GameEnv
from agent import Agent


from collections import  Counter

if __name__ == '__main__':
    env = GameEnv()

    agent = Agent()

    eps = 0.9
    NUM_EPISODES = 10000

    scores = []

    for episode_idx in range(NUM_EPISODES):
        print(f"Progress: {round((episode_idx + 1) * 100 / NUM_EPISODES)}%", end='\r')
        env.reset()
        state = env.get_state()

        # env.print_board()
        while True:
            # Get the action our agent must take at the current state.
            action = agent.act(state, eps)

            # Get the experience vectors.
            state, reward, done, next_state = env.step(action)

            # print(f"action: {action}")
            # env.print_board()

            # Learn from the experience.
            agent.step(state, action, reward, next_state, done)

            # Update next state.
            state = next_state

            # If this episode terminates, move to the next episode.
            if done:
                break

        scores.append(reward)

        eps = max(0.1, eps * 0.95)


    # # Save the trained critic model.
    # torch.save(agent.actor.state_dict(), f'{pathlib.Path(__file__).parent.resolve()}/actor.pt')
    # torch.save(agent.critic.state_dict(), f'{pathlib.Path(__file__).parent.resolve()}/critic.pt')

    # Plot.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    averaged_scores = np.convolve(scores, np.ones(100) / 100, mode='valid')
    plt.plot(np.arange(len(averaged_scores)), averaged_scores, label='avg_100')
    averaged_scores = np.convolve(scores, np.ones(1000) / 1000, mode='valid')
    plt.plot(np.arange(len(averaged_scores)), averaged_scores, label='avg_1000')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import torch

import gymnasium as gym

from agent import Agent


def add_noise(signal, mean, sigma):
    signal += np.random.normal(mean, sigma, signal.shape)
    return signal


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')

    agent = Agent()

    NUM_EPISODES = 5000

    scores = []
    sigma = 0.5

    for episode_idx in range(NUM_EPISODES):
        print(f"Progress: {round((episode_idx + 1) * 100 / NUM_EPISODES)}%", end='\r')
        state, _ = env.reset()

        score = []

        while True:
            # Get the action our agent must take at the current state.
            action = agent.act(state)
            action = add_noise(action, 0., sigma)
            action = np.clip(action, -2., 2.)

            # Get the experience vectors.
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Learn from the experience.
            agent.step(state, action, reward, next_state, done)

            # Update next state.
            state = next_state

            score.append(reward)

            # If this episode terminates, move to the next episode.
            if done:
                break

        scores.append(np.sum(score))

        sigma = max(0.01, sigma * 0.995)

    # Save the trained critic model.
    torch.save(agent.actor.state_dict(), f'{pathlib.Path(__file__).parent.resolve()}/actor.pt')
    torch.save(agent.critic.state_dict(), f'{pathlib.Path(__file__).parent.resolve()}/critic.pt')

    # Plot.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores, label='raw')
    averaged_scores = np.convolve(scores, np.ones(100) / 100, mode='valid')
    plt.plot(np.arange(len(averaged_scores)), averaged_scores, label='avg_100')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.show()

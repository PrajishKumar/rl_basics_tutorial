import gymnasium as gym
import pathlib
import torch

from agent import Agent

env = gym.make('Pendulum-v1', render_mode='human')
state, _ = env.reset()

agent = Agent()
agent.actor.load_state_dict(torch.load(f'{pathlib.Path(__file__).parent.resolve()}/actor.pt'))
agent.critic.load_state_dict(torch.load(f'{pathlib.Path(__file__).parent.resolve()}/critic.pt'))

for i in range(1000):
    action = env.action_space.sample()
    state, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

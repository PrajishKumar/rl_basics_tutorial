"""
Read more: https://www.gymlibrary.dev/environments/classic_control/pendulum/
"""

import gymnasium as gym

env = gym.make('Pendulum-v1', render_mode='human')
state, _ = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    state, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

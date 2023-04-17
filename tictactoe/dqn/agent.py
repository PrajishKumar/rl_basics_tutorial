import torch
import torch.nn as nn
from collections import deque, namedtuple
import numpy as np
import random

BUFFER_SIZE = int(1e6)          # replay buffer size
BATCH_SIZE = 1000               # minibatch size
GAMMA = 0.9                     # discount factor
LR = 5e-2                       # learning rate
UPDATE_EVERY = 10               # how often to update the network


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.stacked_layers = nn.Sequential(
            nn.Linear(9, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 9),
        )

    def forward(self, state):
        return self.stacked_layers(state)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done) -> None:
        experience_tuple = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience_tuple)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # The action-value network that is learnt over time.
        self.q_network = QNetwork().to(self.device)

        # Optimizer for learning the network parameters.
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=LR)

        # Replay buffer.
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device)

        # Initialize time step.
        self.t_step = 0

        # Loss function used.
        self.loss_function = torch.nn.MSELoss()

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory.
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.replay_buffer) > BATCH_SIZE:
                experiences = self.replay_buffer.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        # Epsilon-greedy action selection.
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            self.q_network.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.randint(0, 8)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Action value computed by the Q network.
        action_value_predicted = torch.gather(self.q_network(states), dim=1, index=actions)

        # Expected action value, from the target Q network.
        value_of_next_state = torch.reshape(torch.max(self.q_network(next_states), dim=1)[0], (-1, 1))
        action_value_expected = rewards + gamma * value_of_next_state * (1 - dones)

        # Define the loss function.
        loss = self.loss_function(action_value_predicted, action_value_expected.detach())

        # Update the Q network parameters.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_q_values(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()
        return action_values.cpu().numpy().flatten()

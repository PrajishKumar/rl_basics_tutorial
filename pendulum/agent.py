import torch
import torch.nn as nn
from collections import deque, namedtuple
import random
import numpy as np

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
UPDATE_EVERY = 10  # how often to update the network
TAU = 1e-3  # for soft update of target parameters

LR_ACTOR = 1e-3  # learning rate for the actor
LR_CRITIC = 1e-3  # learning rate for the critic


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.stacked_layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, state):
        return 2.0 * self.stacked_layers(state)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.stacked_layers = nn.Sequential(
            nn.Linear(3 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state, action):
        input_layer = torch.cat((state, action), 1)
        return self.stacked_layers(input_layer)


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

        # The "local" policy network that is learnt over timed.
        self.actor = Actor().to(self.device)

        # The "target" policy network that's updated less frequently to avoid oscillating updates.
        self.actor_target = Actor().to(self.device)

        # Optimizer for learning the parameters of the policy network.
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # The action-value network that is learnt over time.
        self.critic = Critic().to(self.device)

        # The "target" action-value network that's updated less frequently to avoid oscillating updates to Q values.
        self.critic_target = Critic().to(self.device)

        # Optimizer for learning the network parameters.
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Replay buffer.
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device)

        # Initialize time step.
        self.t_step = 0

        # Loss function used.
        self.critic_loss_function = torch.nn.MSELoss()

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

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        return action.reshape(1)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        """ update critic """
        # Predict the next set of actions.
        actions_next = self.actor_target(next_states)

        # Predict the action-value for the next timestep.
        critic_targets_next = self.critic_target(next_states, actions_next)

        # Based on the expected future action-value, compute the expected action-value for the current state.
        critic_target = rewards + (gamma * critic_targets_next * (1 - dones))

        # Compute loss.
        critic_expected = self.critic(states, actions)
        critic_loss = self.critic_loss_function(critic_expected, critic_target.detach())

        # Minimize the loss.
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        """ update actor """
        # Predict the set of actions.
        actions_expected = self.actor(states)

        # Compute loss. We want to favour actions that increase the action-value in that state.
        actor_loss = -self.critic(states, actions_expected).mean()

        # Minimize the loss.
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        """ update target networks """
        self.__update_target_network(self.critic, self.critic_target, TAU)
        self.__update_target_network(self.actor, self.actor_target, TAU)

    @staticmethod
    def __update_target_network(local_model, target_model, tau):
        """
        θ_target = τ * θ_local + (1 - τ) * θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

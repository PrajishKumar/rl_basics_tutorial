from agent import *

TAU = 0.2  # for soft update of target parameters


class AgentWithTargetQNetwork(Agent):
    def __init__(self):
        super(AgentWithTargetQNetwork, self).__init__()

        # The "target" action-value network that's updated less frequently to avoid oscillating updates to Q values.
        self.q_network_target = QNetwork().to(self.device)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Action value computed by the local Q network.
        action_value_predicted = torch.gather(self.q_network(states), dim=1, index=actions)

        # Expected action value, from the target Q network.
        value_of_next_state = torch.reshape(torch.max(self.q_network_target(next_states), dim=1)[0], (-1, 1))
        action_value_expected = rewards + gamma * value_of_next_state * (1 - dones)

        # Define the loss function.
        loss = self.loss_function(action_value_predicted, action_value_expected.detach())

        # Update the Q network parameters.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target Q network.
        self.update_target_q_network(TAU)

    def update_target_q_network(self, tau):
        """
        θ_target = τ * θ_local + (1 - τ) * θ_target
        """
        for target_param, local_param in zip(self.q_network_target.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
